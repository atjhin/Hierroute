from __future__ import annotations

import os
import random
from torch.utils.data import Dataset, Subset, DataLoader, SequentialSampler, WeightedRandomSampler, random_split
from sklearn.model_selection import train_test_split
import torch 
import numpy as np 
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from collections import Counter
from .extra_functions import set_seed
from .hierarchy import Hierarchy

import copy
import torch.nn.functional as F


def _seed_worker(worker_id):
    """
    DataLoader worker_init_fn for num_workers > 0. On Linux's default 'fork' start
    method, worker processes inherit an identical copy of the parent's global
    random/numpy RNG state at fork time -- without reseeding, transforms that read
    from those global RNGs (e.g. RandomBackgroundNoise's `random.random() < p` gate)
    can produce correlated/duplicated decisions across workers. torch already seeds
    its own per-worker RNG uniquely (torch.initial_seed()); this extends that same
    unique-per-worker seed to stdlib random and numpy, which torch doesn't touch.
    """
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def load_excluded_filenames(log_path):

    """
    Parses a Messy_background/moved_images/log.md-style exclusion log and returns the
    set of (class_name, filename) pairs that must be excluded from the base training
    directory, so a held-out messy-background evaluation set never overlaps with what
    the model was trained on.

    Two kinds of log sections are recognised:
      - '### Moved files' (written by move_to_messy_background.py): the (Class,
        Filename) pair naming a file physically moved out of the base directory.
        These are already absent on disk (moved, not copied) by the time training
        runs, but are still included here for a complete, auditable exclusion set.
      - '### Regenerated files' (written by regenerate_messy_background_dirt.py):
        since a moved file's original clean pixels no longer exist, that script
        re-dirtied a *different*, still-present clean image per slot instead -- the
        "Source Clean Filename" column. Those substitute source images ARE still in
        the base directory and must be excluded too, or the model would train on the
        exact same photo it's later evaluated against (dirtied) in Messy_background.

    The log is append-only across script runs, so there may be more than one section
    of either kind. Filenames are not guaranteed unique across different class
    folders in this dataset, so exclusion is scoped per (class, filename) pair, not
    filename alone.

    Args:
        log_path (str): Path to the markdown log file.

    Returns:
        set[tuple[str, str]]: (class_name, filename) pairs to exclude.
    """

    if not os.path.exists(log_path):
        raise FileNotFoundError(
            f"exclude_log_path was set to '{log_path}' but no such file exists. "
            "Refusing to silently skip the exclusion filter."
        )

    with open(log_path) as f:
        content = f.read()

    excluded = set()

    def parse_table_rows(section_marker, filename_col_idx):
        for section in content.split(section_marker)[1:]:
            table_text = section.split('\n## ')[0]
            for line in table_text.splitlines():
                line = line.strip()
                if not line.startswith('|'):
                    continue
                cols = [c.strip() for c in line.strip('|').split('|')]
                if len(cols) <= filename_col_idx:
                    continue
                class_name, filename = cols[0], cols[filename_col_idx]
                if class_name == 'Class' or set(class_name) == {'-'}:
                    continue
                excluded.add((class_name, filename))

    parse_table_rows('### Moved files', filename_col_idx=1)
    parse_table_rows('### Regenerated files', filename_col_idx=2)

    return excluded


class ImageDataset(Dataset):

    """
    A custom PyTorch Dataset for loading and preprocessing image data from a directory
    where each subfolder represents a class.

    This class handles:
    - Class-wise and random sampling with optional size limits
    - Optional image transforms for data augmentation
    - Preprocessing and setup for imbalanced class handling

    Args:
        data_directory (str): Path to the root dataset directory. Each subdirectory should represent a class or another subdirectory to check.
        data_subdirectories (list of str, optional): Subdirectories with additional images, each sub-subdirectory should represent a class.
        class_names (list, optional): List of class names to include. If None, all subdirectories are included.
        class_sizes (list, optional): Number of samples to include per class. If None, uses `max_class_size` for all.
        class_ids (list, optional): Numeric ID for each class (aligned with `class_names`).
        max_class_size (int, optional): Default maximum number of samples to draw per class. Defaults to 10,000.
        image_resolution (int, optional): Final size (height and width) to resize images to. Defaults to 28.
        image_transforms (callable, optional): Image transformations (e.g., data augmentations) to apply. Defaults to None.
        seed (int, optional): Random seed for reproducibility. Defaults to 666.
        class_folder_map (dict, optional): Mapping from a logical class name to a list of
            on-disk folder names.  Use this when a single class spans multiple directories
            (e.g. ``{'Guinardia': ['Guinardia_delicatula', 'Guinardia_flaccida']}``).
            Folders are pooled before the per-class cap is applied. Defaults to ``{}``.
        exclude_log_path (str, optional): Path to a moved-images log (see
            load_excluded_filenames) whose (class, filename) pairs are excluded from
            this dataset -- for keeping a held-out messy-background evaluation set
            (and its regeneration sources) from leaking into training. Defaults to
            None (no exclusion).
        uncapped_subdirectories (list of str, optional): Subdirectories whose images are
            always included in full, added on top of the per-class cap rather than
            competing for it (e.g. a small, newly annotated batch that would otherwise be
            mostly sampled out of a large pooled class). Kept out of the capped pool, so
            the capped sample is unchanged by adding them. Defaults to None.

    Attributes:
        data_directory (str): Path to the dataset root directory.
        data_subdirectories (list of str): Subdirectories with additional images.
        uncapped_subdirectories (list of str): Subdirectories included in full on top of the cap.
        seed (int): Random seed used for sampling.
        class_names (list): Sorted list of class names included in the dataset.
        class_sizes (torch.Tensor): Tensor of the actual sampled size per class.
        class_ids (list): Numeric ID for each class (aligned with `class_names`).
        image_paths (list): List of file paths to all sampled images.
        labels (list): List of numeric class IDs corresponding to each image.
        image_resolution (int): Size to which each image is resized.
        image_transforms (callable or None): Image transformations applied during training or inference.
        format_file (str): Format of images files. Default: '.tif'
        class_folder_map (dict): Mapping from logical class name to disk folder name(s).
        excluded_counts (dict): Number of images excluded per class via exclude_log_path.
    """

    def __init__(self, data_directory, data_subdirectories: list = None, class_names: list = None,
                 class_sizes: list = None, class_ids: list = None, max_class_size: int = 10000,
                 image_resolution: int = 28, image_transforms = None, seed: int = 666,
                 format_file = '.tif', class_folder_map: dict = None, exclude_log_path: str = None,
                 uncapped_subdirectories: list = None):

        self.data_directory = data_directory
        self.data_subdirectories = ['']
        self.uncapped_subdirectories = list(uncapped_subdirectories or [])
        self.seed = seed
        self.class_names = class_names
        self.class_folder_map = class_folder_map or {}
        self.excluded_counts = {}

        excluded_filenames = load_excluded_filenames(exclude_log_path) if exclude_log_path else set()

        set_seed(seed)

        # Additional subdirectories to check
        if data_subdirectories is not None:
            self.data_subdirectories.extend(data_subdirectories)

        # Specify subset of classes to consider; all classes considered if None
        if class_names is None:
            class_names = sorted(os.listdir(self.data_directory))

        # Specify initial number of samples to consider per class; max if None
        if class_sizes is None:
            class_sizes = [max_class_size] * len(class_names)

        # Specify numeric class ID/index per class; in alphabetical order if None
        if class_ids is None:
            class_ids = list(range(len(self.class_names)))
        
        self.class_names, self.class_sizes, self.class_ids = map(
            list, zip(*sorted(zip(class_names, class_sizes, class_ids)))
        )
               
        # Iterate through each class and sample .tif images only; append paths and labels
        self.image_paths = []
        self.labels = []

        for class_id, class_name in zip(self.class_ids, self.class_names):
            
            # Retrieve all image paths across directories for specified class
            class_paths = []

            folder_names = self.class_folder_map.get(class_name, [class_name])

            for data_subdirectory in self.data_subdirectories + self.uncapped_subdirectories:
                uncapped = data_subdirectory in self.uncapped_subdirectories
                for folder_name in folder_names:
                    class_directory = os.path.join(data_directory, data_subdirectory, folder_name)

                    if os.path.isdir(class_directory):
                        class_paths.extend(
                            (folder_name, os.path.join(class_directory, filename), uncapped)
                            for filename in os.listdir(class_directory)
                        )

            if excluded_filenames:
                before_exclusion = len(class_paths)
                # Match against the specific on-disk folder_name each path actually came
                # from, not any folder_name in the pool -- pooled folders can share
                # filenames (e.g. Fiber_Squiggly and Fiber_Hairlike both use the same
                # capture-run naming convention), so matching broadly would wrongly
                # exclude an unrelated same-named file sitting in a sibling folder.
                class_paths = [
                    (folder_name, p, uncapped) for folder_name, p, uncapped in class_paths
                    if (folder_name, os.path.basename(p)) not in excluded_filenames
                ]
                n_excluded = before_exclusion - len(class_paths)
                if n_excluded > 0:
                    print(f'Excluded {n_excluded} images from class "{class_name}" per {exclude_log_path}')
                    self.excluded_counts[class_name] = n_excluded

            uncapped_paths = sorted(p for _, p, uncapped in class_paths if uncapped)
            class_paths = [p for _, p, uncapped in class_paths if not uncapped]

            # Determine new class size and sample images, only include .tif files.
            # Index into self.class_ids (sorted alongside self.class_names/self.class_sizes),
            # not the caller's unsorted class_ids, so each size is stored on its own class.
            class_idx = self.class_ids.index(class_id)
            new_class_size = min(self.class_sizes[class_idx], len(class_paths))

            random.seed(self.seed)
            sampled_paths = random.sample(class_paths, new_class_size)

            # Uncapped subdirectories are added in full on top of the capped sample
            new_class_size += len(uncapped_paths)

            for image_path in sampled_paths + uncapped_paths:
                if image_path.lower().endswith(format_file):
                    try:
                        with Image.open(image_path) as img:
                            img.verify()
                        self.image_paths.append(image_path)
                    except (UnidentifiedImageError, OSError, ValueError):
                        new_class_size -= 1
                else:
                    new_class_size -= 1

            self.class_sizes[class_idx] = new_class_size
            self.labels.extend([class_name] * new_class_size)

        if self.excluded_counts:
            print(f'Total excluded across all classes: {sum(self.excluded_counts.values())}')

        # Other class initializations
        self.image_resolution = image_resolution
        self.image_transforms = image_transforms
        
    
    def __len__(self) -> int:
        """
        Returns the number of samples in the Dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """
        Returns the image and label of specified sample.

        Args:
            idx (int): Index of specified sample.
        """
        
        image = Image.open(self.image_paths[idx]).convert('L')
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.image_transforms:
            image = self.image_transforms(image)

        image = image.repeat(3, 1, 1)
        
        return image, label
            
    def print_dataset_details(self, indices: list = None, subset_name: str = None):

        """
        Prints the class distribution of the Dataset.

        Args:
            indices (list, optional): Specific indices of subset of Dataset to consider.
            subset_name (str, optional): Specific name of subset of Dataset to print.
        """

        if indices is None:
            filtered_labels = self.labels
        else:
            filtered_labels = [self.labels[i] for i in indices]

        filtered_counts = dict(Counter(filtered_labels))

        if subset_name is None:
            print(f'\nTotal Dataset Size: {len(filtered_labels)}')
        else:
            print(f'\n{subset_name} Dataset Size: {len(filtered_labels)}')

        for class_id, class_name in zip(self.class_ids, self.class_names):
            class_prop = filtered_counts[class_name] / len(filtered_labels)

            print(f'Class Name: {class_name} | Class Label: {class_id} | Count: {filtered_counts[class_name]} ' +
                    f'| Prop: {class_prop:.2f}'
                )
            
    def get_dataset_details(self, indices: list = None):

        """
        Get the class distribution of the Dataset.

        Args:
            indices (list, optional): Specific indices of subset of Dataset to consider.
        Output: 
            Dictionary containing: class name, label, count and proportion relative the whole sample    
        """
        
        all_class_names = []
        all_class_labels = []
        all_counts = []
        all_props = []

        if indices is None:
            filtered_labels = self.labels
        else:
            filtered_labels = [self.labels[i] for i in indices]
        # print(f"filtered_labels:{filtered_labels}")
        filtered_counts = dict(Counter(filtered_labels))
        # print(f"filtered_counts:{filtered_counts}")
        for class_id, class_name in zip(self.class_ids, self.class_names):
            try:
                class_prop = filtered_counts[class_name] / len(filtered_labels)
            except:
                print(f"Following class is not found: {class_id}, {class_name}")
            all_class_names.append(class_name)
            all_class_labels.append(class_id)
            all_counts.append(filtered_counts[class_name])
            all_props.append(class_prop)
            
        data_details ={
            'Class': all_class_names,
            'Label': all_class_labels,
            'Counts': all_counts,
            'Prop': all_props
            } 
            
        return data_details
    
    def split_train_test_val(self, train_prop: float = 0.7, val_prop: float = 0.1, test_prop: float = 0.2, verbose: bool = True):

        """
        Returns indices corresponding to the train, validation and test subsets of the Dataset.
        Uses stratified splitting to ensure minority classes are proportionally represented.

        Args:
            train_prop (float): Proportion of samples to allocate to the train subset.
            val_prop (float): Proportion of samples to allocate to the validation subset.
            test_prop (float): Proportion of samples to allocate to the test subset.
            verbose (bool): Specifies whether to print distributions of subsets.
        """

        indices = list(range(len(self)))
        labels = self.labels

        # Split off test set
        train_val_idx, test_idx = train_test_split(
            indices, test_size=test_prop, stratify=labels, random_state=self.seed
        )

        # Split remaining into train and val
        train_val_labels = [labels[i] for i in train_val_idx]
        relative_val = val_prop / (train_prop + val_prop)
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=relative_val, stratify=train_val_labels, random_state=self.seed
        )

        if verbose:
            self.print_dataset_details(train_idx, 'Train')
            self.print_dataset_details(val_idx, 'Validation')
            self.print_dataset_details(test_idx, 'Test')

        return train_idx, val_idx, test_idx
    
    def append_image_transforms(self, image_transforms: transforms.Compose = None, 
                                replace: bool = False, verbose: bool = False):
        
        """
        Appends image transformations to existing transformation pipeline or replaces.
        If multiple `ToTensor()` transformations are included in the resulting pipeline, only the last instance is kept.
        If there are no `ToTensor()` transformations in the resulting pipeline, it is appended.

        Args:
            image_transforms(transfors.Compose, optional): Iterable of image transformations to append.
            replace (bool): Specifies whether to replace with or append the above image_transforms.
            verbose (bool): Specifies whether to print the resulting image transformation pipeline.
        """
        
        if image_transforms is None:
            if self.image_transforms is None:
                image_transforms_list = []
            else: 
                image_transforms_list = self.image_transforms.transforms
        else:
            if replace:
                image_transforms_list = image_transforms.transforms
            else:
                image_transforms_list = self.image_transforms.transforms + image_transforms.transforms

        image_transforms_cleaned = []
        to_tensor_indices = [i for i, tf in enumerate(image_transforms_list) if isinstance(tf, transforms.ToTensor)]

        if to_tensor_indices:
            last_idx = to_tensor_indices[-1]
            image_transforms_cleaned = [tf for i, tf in enumerate(image_transforms_list) if not isinstance(tf, transforms.ToTensor) or i == last_idx]
        else:
            image_transforms_cleaned = image_transforms_list + [transforms.ToTensor()]

        self.image_transforms = transforms.Compose(image_transforms_cleaned)

        if verbose:
            self.print_image_transforms()
            
    def print_image_transforms(self) -> None:
        """
        Prints the ordered image transformations applied to the Dataset.
        """

        print('\nCurrent Image Transform Pipeline:')
        for tf in self.image_transforms.transforms:
            print(' ', tf)    
            
    
    def create_dataloaders(self, batch_size: int, train_indices, val_indices, test_indices,
                           image_transforms: transforms.Compose = None, transform_val: bool = False,
                           train_sample_weights: torch.tensor = None, balanced: bool = False,
                           num_workers: int = 0, pin_memory: bool = True):

        """
        Creates the train, validatinon and test DataLoaders required for training a PyTorch model.
        If `train_sample_weights` is specified, they are supplied to WeightedRandomSampler for the train subset.

        Args:
            batch_size (int): Sizes of batches to process samples in DataLoader.
            train_indices (list): Indices corresponding to the train subset of the Dataset.
            val_indices (list): Indices corresponding to the validation subset of the Dataset.
            test_indices (list): Indices corresponding to the test subset of the Dataset.
            image_transforms (transforms.Compose, optional): Additional image transformations for the train subset.
            transform_val (bool): Specifies whether to apply train image transformations to the validation subset.
            train_sample_weights (torch.tensor, optional): Contains weights for each sample in the train subset.
            balanced (bool): If True and train_sample_weights is None, auto-computes inverse-frequency weights.
            num_workers (int): Subprocesses per DataLoader for parallel data loading/augmentation.
                Defaults to 0 (single-process, loads in the main training loop).
            pin_memory (bool): Use page-locked host memory for faster host->GPU transfer.
        """

        if balanced and train_sample_weights is None:
            train_labels = [self.labels[i] for i in train_indices]
            class_counts = Counter(train_labels)
            train_sample_weights = [1.0 / class_counts[label] for label in train_labels]

        if image_transforms is not None:
            dataset_aug = copy.deepcopy(self)
            dataset_aug.append_image_transforms(
                image_transforms = image_transforms, verbose = False
            )
            train_dataset = Subset(dataset_aug, train_indices)
            if transform_val:
                val_dataset = Subset(dataset_aug, val_indices)
            else:
                val_dataset = Subset(self, val_indices)
        else:
            train_dataset = Subset(self, train_indices)
            val_dataset = Subset(self, val_indices)

        test_dataset = Subset(self, test_indices)

        persistent_workers = num_workers > 0

        if train_sample_weights is None:
            train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True,
                generator = torch.Generator().manual_seed(self.seed),
                num_workers = num_workers, pin_memory = pin_memory,
                persistent_workers = persistent_workers, worker_init_fn = _seed_worker,
            )
        else:
            train_loader = DataLoader(
                train_dataset, batch_size = batch_size,
                sampler = WeightedRandomSampler(train_sample_weights, num_samples = len(train_sample_weights), replacement = True),
                num_workers = num_workers, pin_memory = pin_memory,
                persistent_workers = persistent_workers, worker_init_fn = _seed_worker,
            )

        val_loader = DataLoader(
            val_dataset, batch_size = batch_size, sampler = SequentialSampler(val_dataset),
            num_workers = num_workers, pin_memory = pin_memory,
            persistent_workers = persistent_workers, worker_init_fn = _seed_worker,
        )
        test_loader = DataLoader(
            test_dataset, batch_size = batch_size, sampler = SequentialSampler(test_dataset),
            num_workers = num_workers, pin_memory = pin_memory,
            persistent_workers = persistent_workers, worker_init_fn = _seed_worker,
        )

        return train_loader, val_loader, test_loader
    



from collections import Counter
class HierImageDataset(Dataset):
    """
    A PyTorch Dataset that wraps an :class:`ImageDataset` and attaches a
    :class:`Hierarchy` built from an adjacency graph.

    Each sample returned by ``__getitem__`` contains the image, its one-hot
    label vector (over all hierarchy nodes), the root-to-leaf path, and per-node
    supervision targets/masks used by the loss function.

    Args:
        base_dataset      (ImageDataset):     Source dataset providing image paths and string labels.
        adjacency_graph   (dict[str, list]):  Parent → children mapping that defines the taxonomy tree.
                                              Must contain every class name present in ``base_dataset``.
        levels            (int):              Number of non-root levels in the hierarchy.
        image_transforms  (callable, optional): Transforms applied to images at load time.
        leaves_only       (bool):             If True, samples whose label is an internal
                                              (non-leaf) node are removed after construction.

    Attributes:
        data_directory    (str):        Root directory inherited from ``base_dataset``.
        data_subdirectories (list):     Subdirectories inherited from ``base_dataset``.
        seed              (int):        Random seed inherited from ``base_dataset``.
        image_resolution  (int):        Image resize resolution inherited from ``base_dataset``.
        image_paths       (list[str]):  File paths to all sampled images.
        labels            (list[int]):  Integer node_id for each image (leaf node).
        label_to_ids      (dict):       Mapping from class name (str) to integer node_id.
        hierarchy         (Hierarchy):  Taxonomy tree built from ``adjacency_graph``.
        levels            (int):        Number of non-root hierarchy levels.
        image_transforms  (callable or None): Active image transform pipeline.
    """
       
    def __init__(
        self,
        base_dataset: ImageDataset,
        adjacency_graph: dict[str, list[str]],
        levels: int,
        image_transforms=None,
        leaves_only: bool = False,
    ) -> None:
        
        self.data_directory = base_dataset.data_directory
        self.data_subdirectories = base_dataset.data_subdirectories
        self.seed = base_dataset.seed
        self.image_resolution =  base_dataset.image_resolution
        self.image_paths = base_dataset.image_paths
        self.label_to_ids = {name: i for i, name in enumerate(adjacency_graph.keys())}
        self.levels = levels
        
        self.labels = [self.label_to_ids[label] for label in base_dataset.labels]
        self.class_names = []
        self.class_sizes = []
        self.class_ids = []

        # Build hierarchy object
        self.hierarchy = self._build_hierarchy(adjacency_graph, self.label_to_ids)
        # Other class initializations
        self.image_resolution = base_dataset.image_resolution
        self.image_transforms = image_transforms
        if leaves_only:
            self._filter_leaves()
        
    def __len__(self) -> int:
        """
        Returns the number of samples in the Dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        """
        Load and return a single sample with hierarchical supervision metadata.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict with keys:
                - ``"image"``      (Tensor ``(3, H, W)``): Transformed image (3-channel).
                - ``"label_node"`` (int):                  Integer node_id of the leaf label.
                - ``"path"``       (list[int]):             Root-to-leaf node_id sequence.
                - ``"targets"``    (dict[int, int]):        For each internal node on the path,
                                                            the index of the correct child.
                - ``"masks"``      (dict[int, list[int]]): For each internal node, a list of
                                                            ones with length equal to num_children.
        """
        image = Image.open(self.image_paths[idx]).convert('L')
        if self.image_transforms:
            image = self.image_transforms(image)

        image = image.repeat(3, 1, 1)

        label_node = self.labels[idx]

        # path from root to labeled node
        path = self.hierarchy.get_path_to_root(label_node)

        # supervision per node
        targets = {}
        masks = {}

        for parent in path[:-1]:
            children = self.hierarchy.children(parent)

            target_child = path[path.index(parent) + 1]

            targets[parent] = children.index(target_child)

            masks[parent] = [1] * len(children)

        return {
            "image": image,
            "label_node": label_node,
            "path": path,
            "targets": targets,
            "masks": masks
        }
    
    def _build_hierarchy(self, adjacency_graph: dict[str, list[str]], label_to_ids: dict[str, int]) -> Hierarchy:
        """
        Constructs a Hierarchy object from an adjacency graph.
        
        Args:
            adjacency_graph (dict): Dictionary mapping parent -> list of children
            
        Returns:
            Hierarchy: Populated hierarchy object
        """
        hierarchy = Hierarchy()
        
        all_children = set()
        for _, kids in adjacency_graph.items():
            all_children.update(kids)

        # root = node never appearing as child
        root = [n for n in adjacency_graph if n not in all_children][0]

        hierarchy.add_node(label_to_ids[root], root)

        def dfs(parent):
            for child in adjacency_graph[parent]:
                hierarchy.add_node(label_to_ids[child], child, label_to_ids[parent])
                dfs(child)

        dfs(root)
        
        return hierarchy
    

    def _filter_leaves(self) -> None:
        """
        Filters image_paths and labels to only include samples
        whose label corresponds to a leaf node in the hierarchy.
        """
        leaf_ids = set(
            node_id for node_id in self.hierarchy.nodes
            if self.hierarchy.is_leaf(node_id)
        )

        filtered_paths, filtered_labels = zip(
            *[(path, label) for path, label in zip(self.image_paths, self.labels)
            if label in leaf_ids]
        ) if self.image_paths else ([], [])

        removed = len(self.image_paths) - len(filtered_paths)

        self.image_paths = list(filtered_paths)
        self.labels = list(filtered_labels)

        print(f"[leaves_only] Kept {len(self.image_paths)} samples | Removed {removed} non-leaf samples")

    def collate_fn(self, batch: list) -> dict:
        """
        Custom collate function for use with PyTorch DataLoader.

        Converts a list of per-sample dicts (from ``__getitem__``) into a
        batched dict.  The ``label_node`` field is one-hot encoded over all
        hierarchy nodes so it can be used directly with :meth:`HierRouteNet.loss_fn`.

        Args:
            batch (list[dict]): List of dicts returned by ``__getitem__``.

        Returns:
            dict with keys:
                - ``"image"``      (Tensor ``(B, 3, H, W)``): Stacked image batch.
                - ``"label_node"`` (Tensor ``(B, num_nodes)``): One-hot label vectors.
                - ``"path"``       (list[list[int]]):            Per-sample root-to-leaf paths.
                - ``"targets"``    (list[dict]):                 Per-sample node supervision targets.
                - ``"masks"``      (list[dict]):                 Per-sample node supervision masks.
        """
        image = torch.stack([b["image"] for b in batch])
        label_node = [F.one_hot(torch.tensor(b["label_node"], dtype=torch.long), num_classes=len(self.hierarchy)).float() for b in batch]
        label_node = torch.stack(label_node)
        # label_node = [b["label_node"] for b in batch]
        path = [b["path"] for b in batch]
        targets = [b["targets"] for b in batch]
        masks = [b["masks"] for b in batch]
        return {
            "image": image,
            "label_node": label_node,
            "path": path,
            "targets": targets,
            "masks": masks
        }

    def print_dataset_details(self) -> None:
        """
        Prints the class distribution of the Dataset.
        """
        print(f'\nTotal Dataset: Size = {len(self)} | Levels = {self.levels}')

        # Get all unique labels (leaf nodes) from the dataset
        leaf_counts = dict(Counter(self.labels))
        
        # Calculate counts for all nodes (including parent nodes)
        all_node_counts = {}
        
        # For each node in the hierarchy, calculate its count
        for node_id in self.hierarchy.nodes.keys():
            if self.hierarchy.is_leaf(node_id):
                all_node_counts[node_id] = leaf_counts.get(node_id, 0)
            else:
                # Parent nodes: sum of all descendant leaf counts
                leaf_descendants = self.hierarchy.descendants(node_id) + [node_id]
                count = 0
                for leaf_id in leaf_descendants:
                    count += leaf_counts.get(leaf_id, 0)
                all_node_counts[node_id] = count

        print(f"all_node_counts: {all_node_counts}\n")

        # Group nodes by their depth in hierarchy
        nodes_by_level = {}
        for node_id in self.hierarchy.nodes.keys():
            depth = self.hierarchy.nodes[node_id].depth
            if depth not in nodes_by_level:
                nodes_by_level[depth] = []
            nodes_by_level[depth].append(node_id)

        print(f"nodes_by_level: {nodes_by_level}\n")

        # Print statistics for each level
        for level in range(self.levels + 1):
            if level not in nodes_by_level:
                continue
            
            level_nodes = sorted(nodes_by_level[level], key=lambda x: self.hierarchy.nodes[x].node_id)
            level_counts = {node_id: all_node_counts[node_id] for node_id in level_nodes}
            
            print(f"\n------------------------Level {level}------------------------")
            
            for node_id in level_nodes:
                node = self.hierarchy.nodes[node_id]
                count = all_node_counts[node_id]
                
                # Skip nodes with zero count
                if count == 0:
                    continue
                
                class_prop = count / len(self)
                node_type = "Leaf" if self.hierarchy.is_leaf(node_id) else "Parent"
                
                print(
                    f'Level: {level} | Class Name: {node.name:20s} | Class Label: {node_id:3d} | '
                    f'Type: {node_type:6s} | Count: {count:6d} | Prop: {class_prop:.2f}'
                )
            
            # Check for missing labels (only relevant for leaf nodes at this level)
            leaf_nodes_at_level = [n for n in level_nodes if self.hierarchy.is_leaf(n)]
            if leaf_nodes_at_level:
                missing_count = sum(1 for label in self.labels 
                                if label is None and self.hierarchy.depth(label) == level)
                if missing_count > 0:
                    print(f'\nImportant: At level {level} there are {missing_count} samples with missing labels\n')
                    
    
    def split_train_test_val(
        self,
        train_prop: float = 0.7,
        val_prop: float = 0.1,
        test_prop: float = 0.2,
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Returns indices corresponding to the train, validation and test subsets of the Dataset.
        Uses stratified splitting to ensure minority classes are proportionally represented.

        Args:
            train_prop (float): Proportion of samples to allocate to the train subset.
            val_prop (float): Proportion of samples to allocate to the validation subset.
            test_prop (float): Proportion of samples to allocate to the test subset.
        """

        indices = list(range(len(self)))
        labels = self.labels

        train_val_idx, test_idx = train_test_split(
            indices, test_size=test_prop, stratify=labels, random_state=self.seed
        )

        train_val_labels = [labels[i] for i in train_val_idx]
        relative_val = val_prop / (train_prop + val_prop)
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=relative_val, stratify=train_val_labels, random_state=self.seed
        )

        return train_idx, val_idx, test_idx
    
    def append_image_transforms(
        self, image_transforms: transforms.Compose = None, replace: bool = False
        ):        
        """
        Appends image transformations to existing transformation pipeline or replaces.
        If multiple `ToTensor()` transformations are included in the resulting pipeline, only the last instance is kept.
        If there are no `ToTensor()` transformations in the resulting pipeline, it is appended.

        Args:
            image_transforms(transfors.Compose, optional): Iterable of image transformations to append.
            replace (bool): Specifies whether to replace with or append the above image_transforms.
        """
        
        if image_transforms is None:
            if self.image_transforms is None:
                image_transforms_list = []
            else: 
                image_transforms_list = self.image_transforms.transforms
        else:
            if replace:
                image_transforms_list = image_transforms.transforms
            else:
                image_transforms_list = self.image_transforms.transforms + image_transforms.transforms

        image_transforms_cleaned = []
        to_tensor_indices = [i for i, tf in enumerate(image_transforms_list) if isinstance(tf, transforms.ToTensor)]

        if to_tensor_indices:
            last_idx = to_tensor_indices[-1]
            image_transforms_cleaned = [tf for i, tf in enumerate(image_transforms_list) if not isinstance(tf, transforms.ToTensor) or i == last_idx]
        else:
            image_transforms_cleaned = image_transforms_list + [transforms.ToTensor()]

        self.image_transforms = transforms.Compose(image_transforms_cleaned)
                    
    def create_dataloaders(
        self, batch_size: int, train_indices, val_indices, test_indices,
        image_transforms: transforms.Compose = None, transform_val: bool = False,
        train_sample_weights: torch.tensor = None, balanced: bool = False,
        num_workers: int = 0, pin_memory: bool = True,
    ):

        """
        Creates the train, validatinon and test DataLoaders required for training a PyTorch model.
        If `train_sample_weights` is specified, they are supplied to WeightedRandomSampler for the train subset.

        Args:
            batch_size (int): Sizes of batches to process samples in DataLoader.
            train_indices (list): Indices corresponding to the train subset of the Dataset.
            val_indices (list): Indices corresponding to the validation subset of the Dataset.
            test_indices (list): Indices corresponding to the test subset of the Dataset.
            image_transforms (transforms.Compose, optional): Additional image transformations for the train subset.
            transform_val (bool): Specifies whether to apply train image transformations to the validation subset.
            train_sample_weights (torch.tensor, optional): Contains weights for each sample in the train subset.
            balanced (bool): If True and train_sample_weights is None, auto-computes inverse-frequency weights.
            num_workers (int): Subprocesses per DataLoader for parallel data loading/augmentation.
                Defaults to 0 (single-process, loads in the main training loop).
            pin_memory (bool): Use page-locked host memory for faster host->GPU transfer.
        """

        if balanced and train_sample_weights is None:
            train_labels = [self.labels[i] for i in train_indices]
            class_counts = Counter(train_labels)
            train_sample_weights = [1.0 / class_counts[label] for label in train_labels]

        if image_transforms is not None:
            dataset_aug = copy.deepcopy(self)
            dataset_aug.append_image_transforms(
                image_transforms = image_transforms, verbose = False
            )
            train_dataset = Subset(dataset_aug, train_indices)
            if transform_val:
                val_dataset = Subset(dataset_aug, val_indices)
            else:
                val_dataset = Subset(self, val_indices)
        else:
            train_dataset = Subset(self, train_indices)
            val_dataset = Subset(self, val_indices)

        test_dataset = Subset(self, test_indices)

        persistent_workers = num_workers > 0

        if train_sample_weights is None:
            train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True,
                collate_fn=self.collate_fn, generator = torch.Generator().manual_seed(self.seed),
                num_workers = num_workers, pin_memory = pin_memory,
                persistent_workers = persistent_workers, worker_init_fn = _seed_worker,
            )
        else:
            train_loader = DataLoader(
                train_dataset, batch_size = batch_size, collate_fn=self.collate_fn,
                sampler = WeightedRandomSampler(train_sample_weights, num_samples = len(train_sample_weights), replacement = True),
                num_workers = num_workers, pin_memory = pin_memory,
                persistent_workers = persistent_workers, worker_init_fn = _seed_worker,
            )

        val_loader = DataLoader(
            val_dataset, batch_size = batch_size, sampler = SequentialSampler(val_dataset),
            collate_fn=self.collate_fn,
            num_workers = num_workers, pin_memory = pin_memory,
            persistent_workers = persistent_workers, worker_init_fn = _seed_worker,
        )
        test_loader = DataLoader(
            test_dataset, batch_size = batch_size, sampler = SequentialSampler(test_dataset),
            collate_fn=self.collate_fn,
            num_workers = num_workers, pin_memory = pin_memory,
            persistent_workers = persistent_workers, worker_init_fn = _seed_worker,
        )

        return train_loader, val_loader, test_loader
        
