"""
flowcam.py — Reading FlowCam exports and cropping individual particles.

A FlowCam export is a set of mosaic images (``<run>_NNNNNN.tif``) plus one or more CSVs with
one row per particle. Each row's ``Image File`` names the mosaic the particle sits on, and
``Image X/Y/Width/Height`` give its box. CSVs may sit next to the mosaics or in a separate
folder, at any depth; they are matched to mosaics through the ``Image File`` column, not by
file name.

Crops are named ``<mosaic stem>_<Particle ID>.tif`` and cut with the box
``(Image X, Image Y, Image X + Image Width, Image Y + Image Height)`` -- the same convention as
HierNet's ``Zooplankton-MNR/pre_process_data.ipynb``. Boxes past the mosaic edge are
zero-padded (PIL's behaviour) and flagged.
"""
from __future__ import annotations

import csv
import io
import multiprocessing
import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

REQUIRED_COLUMNS = ("Particle ID", "Image File", "Image X", "Image Y", "Image Width", "Image Height")
BOX_COLUMNS = ["Image X", "Image Y", "Image Width", "Image Height"]
MOSAIC_NUMBER = re.compile(r"_\d{6}$")


def csv_column_to_snake_case(name: str) -> str:
    """FlowCam column name -> ``csv_<snake_case>`` (e.g. ``'Area (ABD)'`` -> ``'csv_area_abd'``)."""
    return "csv_" + re.sub(r"\s+", "_", name.replace("(", "").replace(")", "").strip().lower())


def run_name_from_image_file(image_file: str) -> str:
    """``'<run>_000012.tif'`` -> ``'<run>'``."""
    return MOSAIC_NUMBER.sub("", Path(image_file).stem)


def run_name_from_crop(crop_name: str) -> str:
    """``'<run>_000012_34.tif'`` -> ``'<run>'``; names that don't follow the convention are returned unchanged."""
    stem = Path(crop_name).stem
    parts = stem.rsplit("_", 2)
    if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
        return parts[0]
    return stem


# ---------------------------------------------------------------------------
# CSV reading
# ---------------------------------------------------------------------------
def is_flowcam_csv(path: str | Path) -> bool:
    """True if the CSV's header has an ``Image File`` column."""
    try:
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            header = next(csv.reader(f), [])
    except OSError:
        return False
    return "Image File" in header


def _has_unquoted_commas(path: Path) -> bool:
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        n_names = len([h for h in header if h != ""])
        return any(len(row) > n_names for row in reader)


def _read_with_unquoted_commas(path: Path) -> pd.DataFrame:
    """FlowCam writes 'Image File' unquoted, so a run named with a comma splits that value across
    two fields and shifts every later column. Rejoin the surplus fields into 'Image File'."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        while header and header[-1] == "":
            header.pop()
        image_file_idx = header.index("Image File")
        rows = []
        for row in reader:
            extra = len(row) - len(header)
            if extra > 0:
                row = row[:image_file_idx] + [",".join(row[image_file_idx:image_file_idx + extra + 1])] \
                    + row[image_file_idx + extra + 1:]
            rows.append(row)
    df = pd.DataFrame(rows, columns=header)
    return pd.read_csv(io.StringIO(df.to_csv(index=False)))


def read_flowcam_csv(path: str | Path) -> pd.DataFrame:
    """Read one FlowCam export CSV, repairing unquoted commas and dropping trailing empty columns."""
    path = Path(path)
    df = _read_with_unquoted_commas(path) if _has_unquoted_commas(path) else pd.read_csv(path)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required FlowCam column(s) {missing}")
    bad = ~df["Image File"].astype(str).str.fullmatch(r".+_\d{6}\.tif")
    if bad.any():
        raise ValueError(f"{path}: {int(bad.sum())} rows have a malformed Image File, "
                         f"e.g. {df.loc[bad, 'Image File'].iloc[0]!r}")
    return df


def find_flowcam_csvs(*roots: str | Path) -> list[Path]:
    """Every CSV under ``roots`` (recursively) whose header has an ``Image File`` column."""
    found = set()
    for root in roots:
        root = Path(root)
        candidates = [root] if root.is_file() else root.rglob("*")
        found.update(p for p in candidates if p.suffix.lower() == ".csv" and is_flowcam_csv(p))
    return sorted(found)


def is_mosaic_name(name: str) -> bool:
    """FlowCam mosaic file names: ``<run>_NNNNNN.tif`` (not ``_bin`` masks or ``cal_image`` files)."""
    stem, ext = os.path.splitext(name)
    return ext.lower() == ".tif" and bool(MOSAIC_NUMBER.search(stem)) and not stem.startswith("cal_image")


def find_mosaics(root: str | Path) -> dict[str, list[Path]]:
    """Mosaic file name -> every path with that name under ``root``."""
    mosaics: dict[str, list[Path]] = {}
    for path in Path(root).rglob("*.tif"):
        if is_mosaic_name(path.name):
            mosaics.setdefault(path.name, []).append(path)
    return mosaics


def _closest(paths: list[Path], anchor: Path) -> Path:
    """The path sharing the longest common directory prefix with ``anchor``."""
    def shared(p: Path) -> int:
        return len(os.path.commonpath([str(p.parent), str(anchor.parent)]))
    return max(paths, key=shared)


# ---------------------------------------------------------------------------
# Class labels
# ---------------------------------------------------------------------------
@dataclass
class ClassRules:
    """
    How a CSV row's label becomes a class (folder) name.

    Attributes:
        label_column     (str):       CSV column holding the manual class (FlowCam: ``Class``).
        append_group_for (list[str]): Classes renamed to ``<class>_<group>`` using the group column,
                                      e.g. ``Nauplius`` -> ``Nauplius_Copepod``.
        group_column     (str):       Column with the group, matched case-insensitively.
        default_group    (str):       Group used when the CSV has no group column.
        rename           (dict):      Final class -> replacement name, applied after the group rule.
        ignore           (set[str]):  Classes skipped entirely (after renaming).
    """
    label_column: str = "Class"
    append_group_for: list[str] = field(default_factory=lambda: ["Nauplius"])
    group_column: str = "Group"
    default_group: str = "Copepod"
    rename: dict[str, str] = field(default_factory=dict)
    ignore: set[str] = field(default_factory=set)

    def labels(self, df: pd.DataFrame) -> pd.Series:
        """Class name per row; NaN for rows whose label is missing."""
        if self.label_column not in df.columns:
            raise ValueError(f"Label column '{self.label_column}' not found (columns: {list(df.columns)[:12]} ...)")
        labels = df[self.label_column].astype("string")
        if self.append_group_for:
            group_col = next((c for c in df.columns if c.lower() == self.group_column.lower()), None)
            groups = df[group_col].astype("string") if group_col else pd.Series(self.default_group, index=df.index)
            needs_group = labels.isin(self.append_group_for)
            labels = labels.where(~needs_group, labels + "_" + groups.fillna(self.default_group))
        if self.rename:
            labels = labels.replace(self.rename)
        return labels


# ---------------------------------------------------------------------------
# Particle table
# ---------------------------------------------------------------------------
@dataclass
class ParticleTable:
    """
    One row per particle across every matched CSV.

    ``particles`` columns: image_file (crop name), run, mosaic_path, csv_file, particle_id,
    x, y, w, h, label (when labelled), status, plus every other CSV column as ``csv_*``.
    status is ``ok``, ``missing mosaic``, ``zero-size box``, ``missing label`` or ``ignored class``.
    """
    particles: pd.DataFrame
    n_csvs: int
    n_mosaics: int
    unused_mosaics: list[str]

    def ok(self) -> pd.DataFrame:
        return self.particles[self.particles["status"] == "ok"]

    def summary(self) -> str:
        counts = self.particles["status"].value_counts().to_dict()
        lines = [f"{len(self.particles)} particles from {self.n_csvs} CSV(s) across {self.n_mosaics} mosaic(s)"]
        lines += [f"  {status}: {n}" for status, n in counts.items()]
        if self.unused_mosaics:
            lines.append(f"  mosaics with no CSV rows: {len(self.unused_mosaics)} (e.g. {self.unused_mosaics[0]})")
        return "\n".join(lines)


def load_particles(export_dir: str | Path, csv_dir: str | Path | None = None,
                   rules: ClassRules | None = None) -> ParticleTable:
    """
    Build the particle table for a FlowCam export.

    Args:
        export_dir: Folder searched recursively for mosaics (and for CSVs, if ``csv_dir`` is None).
        csv_dir:    Folder (or single CSV) holding the export CSVs, if they live elsewhere.
        rules:      Class labelling rules; None for unlabelled data.
    """
    export_dir = Path(export_dir)
    csv_paths = find_flowcam_csvs(csv_dir if csv_dir is not None else export_dir)
    if not csv_paths:
        raise FileNotFoundError(f"No FlowCam CSVs (with an 'Image File' column) found under {csv_dir or export_dir}")
    mosaics = find_mosaics(export_dir)

    tables = []
    for csv_path in csv_paths:
        df = read_flowcam_csv(csv_path)
        table = pd.DataFrame({
            "image_file": df["Image File"].str.removesuffix(".tif") + "_" + df["Particle ID"].astype(int).astype(str) + ".tif",
            "run": df["Image File"].map(run_name_from_image_file),
            "mosaic_name": df["Image File"],
            "csv_file": str(csv_path),
            "particle_id": df["Particle ID"].astype(int),
            "x": df["Image X"], "y": df["Image Y"], "w": df["Image Width"], "h": df["Image Height"],
        })
        if rules is not None:
            table["label"] = rules.labels(df).values
        drop = ["Particle ID", "Image File"] + ([rules.label_column] if rules is not None else [])
        metadata = df.drop(columns=drop).rename(columns=csv_column_to_snake_case)
        tables.append(pd.concat([table, metadata.reset_index(drop=True)], axis=1))

    particles = pd.concat(tables, ignore_index=True)
    # The same particle can appear in more than one export CSV (e.g. a partial and a full export).
    key = ["mosaic_name", "particle_id"] + (["label"] if rules is not None else [])
    particles = particles.drop_duplicates(subset=key, keep="first").reset_index(drop=True)

    particles["mosaic_path"] = [
        str(_closest(mosaics[name], Path(csv_file))) if name in mosaics else None
        for name, csv_file in zip(particles["mosaic_name"], particles["csv_file"])
    ]
    status = np.full(len(particles), "ok", dtype=object)
    status[(particles["w"] <= 0) | (particles["h"] <= 0)] = "zero-size box"
    status[particles["mosaic_path"].isna()] = "missing mosaic"
    if rules is not None:
        status[particles["label"].isna()] = "missing label"
        status[particles["label"].isin(list(rules.ignore))] = "ignored class"
    particles["status"] = status

    unused = sorted(set(mosaics) - set(particles["mosaic_name"]))
    return ParticleTable(particles=particles, n_csvs=len(csv_paths), n_mosaics=len(mosaics), unused_mosaics=unused)


def boxes_out_of_bounds(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    return (boxes[:, 0] + boxes[:, 2] > width) | (boxes[:, 1] + boxes[:, 3] > height) \
        | (boxes[:, 0] < 0) | (boxes[:, 1] < 0)


def crop_box(mosaic: Image.Image, box) -> Image.Image:
    x, y, w, h = (int(v) for v in box)
    return mosaic.crop((x, y, x + w, y + h))


# ---------------------------------------------------------------------------
# Writing crops to disk
# ---------------------------------------------------------------------------
def _crop_one_mosaic(job):
    mosaic_path, row_ids, boxes, out_paths, overwrite = job
    results = []
    try:
        with Image.open(mosaic_path) as mosaic:
            mosaic.load()
            width, height = mosaic.size
            oob = boxes_out_of_bounds(boxes, width, height)
            for row_id, box, out_path, out_of_bounds in zip(row_ids, boxes, out_paths, oob):
                if os.path.exists(out_path) and not overwrite:
                    results.append((row_id, "exists", bool(out_of_bounds)))
                    continue
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                crop_box(mosaic, box).save(out_path)
                results.append((row_id, "written", bool(out_of_bounds)))
    except OSError:
        results = [(row_id, "unreadable mosaic", False) for row_id in row_ids]
    return results


def write_crops(table: ParticleTable, out_dir: str | Path, by: str = "label",
                overwrite: bool = False, workers: int = 4) -> pd.DataFrame:
    """
    Crop every ``ok`` particle to ``out_dir/<label or run>/<image_file>`` and return the
    manifest (the particle table plus ``crop_path``, ``crop_status`` and ``out_of_bounds``).

    Args:
        by: ``"label"`` (one folder per class) or ``"run"`` (one folder per FlowCam run).
    """
    out_dir = Path(out_dir)
    particles = table.particles.copy()
    particles["crop_path"] = None
    particles["crop_status"] = particles["status"]
    particles["out_of_bounds"] = False

    ok = particles["status"] == "ok"
    folder = particles.loc[ok, by].astype(str)
    particles.loc[ok, "crop_path"] = [str(out_dir / f / name) for f, name in zip(folder, particles.loc[ok, "image_file"])]

    jobs = []
    for mosaic_path, group in particles[ok].groupby("mosaic_path", sort=True):
        jobs.append((mosaic_path, group.index.to_numpy(), group[["x", "y", "w", "h"]].to_numpy(dtype=np.int64),
                     group["crop_path"].tolist(), overwrite))

    if workers > 1 and len(jobs) > 1:
        # spawn, not fork: forking a process that has already started torch's threads can deadlock.
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            outcomes = [r for chunk in pool.map(_crop_one_mosaic, jobs, chunksize=8) for r in chunk]
    else:
        outcomes = [r for job in jobs for r in _crop_one_mosaic(job)]

    for row_id, crop_status, out_of_bounds in outcomes:
        particles.at[row_id, "crop_status"] = crop_status
        particles.at[row_id, "out_of_bounds"] = out_of_bounds
    return particles


# ---------------------------------------------------------------------------
# In-memory cropping for inference
# ---------------------------------------------------------------------------
class MosaicCropDataset(Dataset):
    """
    One item per mosaic: every particle on it, cropped in memory and transformed.

    Opening each mosaic once and cropping all its particles is far cheaper than one open per
    particle. Items are ``(images (N, 3, H, W) or None, row_indices, out_of_bounds or None)``;
    ``None`` images mean the mosaic could not be read. Use :func:`collate_mosaics` as collate_fn.
    """

    def __init__(self, particles: pd.DataFrame, transform):
        self.transform = transform
        self.items = [
            (mosaic_path, group.index.to_numpy(), group[["x", "y", "w", "h"]].to_numpy(dtype=np.int64))
            for mosaic_path, group in particles.groupby("mosaic_path", sort=True)
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        mosaic_path, row_indices, boxes = self.items[idx]
        try:
            mosaic = Image.open(mosaic_path).convert("L")
        except OSError:
            return None, row_indices, None
        width, height = mosaic.size
        crops = [self.transform(crop_box(mosaic, box)) for box in boxes]
        return torch.stack(crops).repeat(1, 3, 1, 1), row_indices, boxes_out_of_bounds(boxes, width, height)


def collate_mosaics(batch):
    loaded = [b for b in batch if b[0] is not None]
    missing = [b[1] for b in batch if b[0] is None]
    if not loaded:
        return None, None, None, missing
    images = torch.cat([b[0] for b in loaded])
    row_indices = np.concatenate([b[1] for b in loaded])
    out_of_bounds = np.concatenate([b[2] for b in loaded])
    return images, row_indices, out_of_bounds, missing
