"""
config.py — Hierarchy definitions and training configuration files.

A hierarchy YAML file defines one taxonomy:

    name: mnr_17leaf
    tree:                 # parent -> ordered children; key order defines node IDs
      root: [Zoop-yes, Zoop-No]
      ...
    classes: [...]        # optional: training class names (default: every leaf)
    folder_map:           # optional: class -> on-disk folders pooled into it for training
      Bosmina: [Bosminidae, Bosmina_1, Eubosmina, Bosmina]
    eval_folder_map:      # optional: extra on-disk folder -> node used when grading
      Detritus: Zoop-No

Bundled hierarchies live in ``hierroute/hierarchies/`` and are referenced by name
(``mnr_17leaf``, ``mnr_13leaf``, ``whoi_s``, ``whoi_l``); any other value is treated
as a path to a YAML file.

A training config YAML file is merged over :data:`TRAIN_DEFAULTS`; see
``configs/`` in the repository for complete examples.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import yaml

from .hierarchy import Hierarchy

BUNDLED_HIERARCHIES = ("mnr_17leaf", "mnr_13leaf", "whoi_s", "whoi_l")


@dataclass
class HierarchySpec:
    """
    A taxonomy plus the folder-name conventions used to train and grade against it.

    Attributes:
        name            (str):                  Short identifier, e.g. ``"mnr_17leaf"``.
        tree            (dict[str, list[str]]): Parent -> ordered children. Key order defines node IDs.
        classes         (list[str] | None):     Training class names; ``None`` means every leaf.
        folder_map      (dict[str, list[str]]): Class -> on-disk folder names pooled into it for training.
                                                A listed class only reads the folders given, not its own name.
        eval_folder_map (dict[str, str]):       Extra on-disk folder -> node name used when grading
                                                labelled data (on top of the leaf names and folder_map).
    """
    name: str
    tree: dict[str, list[str]]
    classes: list[str] | None = None
    folder_map: dict[str, list[str]] = field(default_factory=dict)
    eval_folder_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        nodes = set(self.tree)
        children = [c for kids in self.tree.values() for c in kids]
        undefined = set(children) - nodes
        if undefined:
            raise ValueError(f"Hierarchy '{self.name}': children with no entry of their own: {sorted(undefined)}")
        duplicated = {c for c in children if children.count(c) > 1}
        if duplicated:
            raise ValueError(f"Hierarchy '{self.name}': nodes with more than one parent: {sorted(duplicated)}")
        roots = [n for n in self.tree if n not in set(children)]
        if len(roots) != 1:
            raise ValueError(f"Hierarchy '{self.name}': expected exactly one root, found {roots}")
        for label, values in (("classes", self.classes or []), ("folder_map", self.folder_map),
                              ("eval_folder_map", self.eval_folder_map.values())):
            unknown = set(values) - nodes
            if unknown:
                raise ValueError(f"Hierarchy '{self.name}': {label} refers to unknown nodes: {sorted(unknown)}")

    @property
    def leaves(self) -> list[str]:
        """Leaf node names, in tree key order."""
        return [name for name, kids in self.tree.items() if not kids]

    @property
    def training_classes(self) -> list[str]:
        """Class names read from disk for training (``classes`` if set, else every leaf)."""
        return list(self.classes) if self.classes is not None else self.leaves

    def build(self) -> tuple[Hierarchy, dict[str, int]]:
        """Return ``(hierarchy, label_to_id)`` for this tree."""
        return Hierarchy.from_adjacency(self.tree)

    def folder_to_node(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        """
        On-disk folder name -> node name, for grading labelled data.

        Built from, in increasing priority: each leaf's own name, ``folder_map`` (a class's
        listed folders), ``eval_folder_map``, then ``extra``.
        """
        mapping = {leaf: leaf for leaf in self.leaves}
        for node, folders in self.folder_map.items():
            for folder in folders:
                mapping[folder] = node
        mapping.update(self.eval_folder_map)
        if extra:
            unknown = set(extra.values()) - set(self.tree)
            if unknown:
                raise ValueError(f"Folder map refers to nodes not in hierarchy '{self.name}': {sorted(unknown)}")
            mapping.update(extra)
        return mapping

    def to_dict(self) -> dict:
        out = {"name": self.name, "tree": self.tree}
        if self.classes is not None:
            out["classes"] = self.classes
        if self.folder_map:
            out["folder_map"] = self.folder_map
        if self.eval_folder_map:
            out["eval_folder_map"] = self.eval_folder_map
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "HierarchySpec":
        missing = {"name", "tree"} - set(data)
        if missing:
            raise ValueError(f"Hierarchy definition is missing {sorted(missing)}")
        tree = {str(k): [str(c) for c in (v or [])] for k, v in data["tree"].items()}
        return cls(
            name=str(data["name"]),
            tree=tree,
            classes=data.get("classes"),
            folder_map={k: list(v) for k, v in (data.get("folder_map") or {}).items()},
            eval_folder_map=dict(data.get("eval_folder_map") or {}),
        )


def load_hierarchy(name_or_path: str | Path) -> HierarchySpec:
    """Load a bundled hierarchy by name, or a hierarchy YAML file by path."""
    if str(name_or_path) in BUNDLED_HIERARCHIES:
        text = resources.files("hierroute").joinpath("hierarchies", f"{name_or_path}.yaml").read_text()
    else:
        path = Path(name_or_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Hierarchy '{name_or_path}' is neither a bundled hierarchy {BUNDLED_HIERARCHIES} nor a file"
            )
        text = path.read_text()
    return HierarchySpec.from_dict(yaml.safe_load(text))


def load_folder_map(path: str | Path) -> dict[str, str]:
    """Load a grading folder map YAML file: ``{on-disk folder: node name}``."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping of folder name -> node name")
    return {str(k): str(v) for k, v in data.items()}


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------
TRAIN_DEFAULTS: dict = {
    "hierarchy": "mnr_17leaf",
    "model": {
        "backbone": "swin_t",          # swin_t | swin_s | efficientnet_b0
        "expert": "cnn",               # cnn | mlp | linear
        "loss": "bce",                 # bce | focal
        "focal_gamma": 2.0,
        "freeze_backbone": False,
        "pretrained_backbone": True,   # start from torchvision's ImageNet weights (downloaded once)
    },
    "data": {
        "data_dir": None,              # required: folder with one subfolder per class
        "subdirectories": [],          # extra folders under data_dir with the same layout
        "uncapped_subdirectories": [], # always included in full, on top of max_class_size
        "exclude_log": None,           # moved-images log.md whose files are excluded (relative to data_dir)
        "format": ".tif",
        "resolution": 64,
        "max_class_size": 12000,
        "split": [0.7, 0.1, 0.2],      # train / validation / test
        "balanced_sampling": True,
    },
    "augment": {
        "background_noise": True,
        "noise_p": 0.8,
        "hflip": True,
        "vflip": True,
    },
    "train": {
        "lr": 3.0e-4,
        "max_epochs": 40,
        "scheduler": "cosine",         # cosine | none
        "patience": 10,
        "delta": 0.001,
        "gradient_clip": 1.0,
        "batch_size": 64,
        "num_workers": 8,
        "seed": 666,
        "device": "auto",              # auto | cuda | cpu | cuda:N
    },
    "out_dir": "training_result",
    "run_name": None,                  # default: <backbone>_<expert>_<noise|baseline>_<MM-DD>
}


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``; unknown keys raise."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key not in merged:
            raise KeyError(f"Unknown config key '{key}' (expected one of {sorted(merged)})")
        if isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_section(merged[key], value, key)
        else:
            merged[key] = value
    return merged


def _merge_section(base: dict, override: dict, section: str) -> dict:
    unknown = set(override) - set(base)
    if unknown:
        raise KeyError(f"Unknown config key(s) in '{section}': {sorted(unknown)} (expected {sorted(base)})")
    return {**base, **override}


def set_dotted(config: dict, dotted_key: str, raw_value: str) -> None:
    """Apply a ``section.key=value`` override; the value is parsed as YAML (numbers, lists, null)."""
    keys = dotted_key.split(".")
    target = config
    for key in keys[:-1]:
        if key not in target or not isinstance(target[key], dict):
            raise KeyError(f"Unknown config section '{key}' in '{dotted_key}'")
        target = target[key]
    if keys[-1] not in target:
        raise KeyError(f"Unknown config key '{dotted_key}'")
    target[keys[-1]] = yaml.safe_load(raw_value)


def load_train_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Load a training config file merged over :data:`TRAIN_DEFAULTS`, then apply ``overrides``."""
    config = copy.deepcopy(TRAIN_DEFAULTS)
    if path is not None:
        config = deep_merge(config, yaml.safe_load(Path(path).read_text()) or {})
    if overrides:
        config = deep_merge(config, overrides)
    return config
