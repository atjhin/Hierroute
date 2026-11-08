"""
hierroute — Hierarchical Mixture-of-Experts for image classification.

Public API:
    Data:      ImageDataset, HierImageDataset
    Hierarchy: Node, Hierarchy, HierarchySpec, load_hierarchy
    Model:     Expert, HierRouteNet
    Training:  Trainer
    Inference: load_model, ModelConfig, evaluate, EvaluateOptions
    Utilities: set_seed, Visualize, visualize_swin_attention
    Constants: ZOOPLANKTON_CLASSES, hier_adjacency_graph, hiernet_adjacency_graph,
               whoi_adjacency_graph_s, whoi_adjacency_graph_l, SEED, MAX_CLASS_SIZE, RESOLUTION, LEVELS

Command line: ``hierroute crop | evaluate | train | download`` (see ``hierroute -h``).
"""
__version__ = "0.1.0"

from .model import Expert, HierRouteNet
from .trainer import Trainer
from .data_setup import ImageDataset, HierImageDataset
from .hierarchy import Node, Hierarchy
from .config import HierarchySpec, load_hierarchy
from .pretrained import ModelConfig, load_model
from .evaluate import EvaluateOptions, evaluate
from .extra_functions import set_seed, Visualize, visualize_swin_attention
from .constants import *
