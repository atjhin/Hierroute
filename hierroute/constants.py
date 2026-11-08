"""
constants.py — Project-wide configuration values and taxonomy adjacency graphs.

Adjacency graphs:
    hier_adjacency_graph    : 3-level freshwater zooplankton taxonomy (Zooplankton-MNR dataset),
                              a simplified 13-leaf version.
    hiernet_adjacency_graph : Full-resolution freshwater zooplankton taxonomy, mirroring
                              HierNet's Zooplankton-MNR/config.yml hierarchy.level_definitions
                              exactly (17 leaves).
    whoi_adjacency_graph_l : 5-level marine diatom taxonomy based on phylogenetic lineage
                             (Plankton-WHOI dataset, long/deep hierarchy variant).
    whoi_adjacency_graph_s : 3-level marine diatom taxonomy based on morphological traits
                             (Plankton-WHOI dataset, short/shallow hierarchy variant).
"""

ZOOPLANKTON_CLASSES = [
     'Bubbles',
     'Exoskeleton',
     'Plant_Matter',
     'Fiber_Squiggly',
     'Fiber_Hairlike',
     'Copepoda',
     'Calanoid',
     'Cyclopoid',
     'Harpacticoid',
     'Cladocera',
     'Bosminidae',
     'Daphnia',
     'Rotifer',
     'Nauplius_Copepod',
     'Diaphonosoma'
]

# 'Sididae' is an on-disk folder of Diaphonosoma specimens under a legacy/alternate
# name -- pooled into the 'Diaphonosoma' class via ImageDataset's class_folder_map.
MNR_CLASS_FOLDER_MAP = {
    'Diaphonosoma': ['Diaphonosoma', 'Sididae'],
}

import os
data_directory = os.environ.get('ZOOPLANKTON_DATA_DIR', 'data')
data_subdirectories = ['Processed Data']
SEED = 666            # Global reproducibility seed

MAX_CLASS_SIZE = 12000  # Maximum samples drawn per class during dataset construction
RESOLUTION = 64        # Target image resolution in pixels (H × W)

LEVELS = 3             # Number of non-root levels in the MNR hierarchy

# ---------------------------------------------------------------------------
# Zooplankton-MNR hierarchy (freshwater, Ontario Ministry of Natural Resources)
# 19 nodes | 3 levels | 13 leaf classes
# ---------------------------------------------------------------------------
hier_adjacency_graph = {
    'root': ['Zoop-yes', 'Zoop-No'],

    'Zoop-yes': ['Copepoda', 'Cladocera', 'Rotifer'],
    'Zoop-No': ['Bubbles', 'Exoskeleton', 'Fiber', 'Plant_Matter'],

    'Copepoda': ['Cyclopoid', 'Calanoid', 'Harpacticoid', 'Nauplius_Copepod'],
    'Cladocera': ['Bosminidae', 'Daphnia', 'Diaphonosoma'],

    'Fiber': ['Fiber_Hairlike', 'Fiber_Squiggly'],
    'Rotifer': [],
    'Bubbles': [],
    'Exoskeleton': [],
    'Plant_Matter': [],
    'Fiber_Squiggly': [],
    'Fiber_Hairlike': [],
    'Cyclopoid': [],
    'Calanoid': [],
    'Harpacticoid': [],
    'Nauplius_Copepod': [],
    'Bosminidae': [],
    'Daphnia': [],
    'Diaphonosoma': []
}

# ---------------------------------------------------------------------------
# HierNet Zooplankton-MNR hierarchy -- mirrors HierNet's
# Zooplankton-MNR/config.yml `hierarchy.level_definitions` exactly (finest ->
# coarsest: 17 level-0 coarse_names, 7 level-1, 2 level-2 Zoop-yes/Zoop-No).
#
# Mixed leaf depth by design, matching config.yml: 'Bubbles', 'Exoskeleton',
# 'Fiber' and 'Plant_Matter' have identical membership at both level-0 and
# level-1 in config.yml (no finer sub-classification exists for them), so they
# sit as leaves directly under 'Zoop-No' rather than at the deepest level --
# unlike e.g. 'Copepoda', whose level-0 groups (Cyclopoid/Calanoid/...) are
# strictly finer than its own level-1 label. HierRouteNet's routing/loss are
# purely path-based and don't require uniform leaf depth.
#
# Raw folders in config.yml's `data.classes` that are self-referential coarse
# labels with no distinct level-0 sub-group (generic Copepoda/Cladocera/Rotifer
# folders) have no leaf home in a strict tree and are intentionally omitted --
# same treatment hier_adjacency_graph already gives Copepoda/Cladocera. Debris
# has no leaf home either, but (like Bubbles/Exoskeleton/Fiber_*/Plant_Matter)
# is pooled into 'Zoop-No' via HIERNET_CLASS_FOLDER_MAP below.
#
# 24 nodes | 3 levels (mixed leaf depth 2-3) | 17 leaf classes
# ---------------------------------------------------------------------------
hiernet_adjacency_graph = {
    'root': ['Zoop-yes', 'Zoop-No'],

    'Zoop-yes': ['Copepoda', 'Cladocera', 'Rotifer', 'Zoop_Unknown'],
    # 'Zoop-No': ['Bubbles', 'Exoskeleton', 'Fiber', 'Plant_Matter'],
    'Zoop-No': [],

    'Copepoda': ['Cyclopoid', 'Calanoid',
                #  'Harpacticoid',
                 'Nauplius_Copepod', 'Copepoda_Unknown'],
    'Cladocera': ['Bosmina', 'Daphnia', 'Holopedium', 'Chydorus', 'Diaphonosoma', 'Cladocera_Unknown'],
    'Rotifer': ['Trichocerca', 'Asplanchna', 'Conochilus', 'Kellicottia', 'Rotifer_Unknown'],

    # 'Bubbles': [],
    # 'Exoskeleton': [],
    # 'Fiber': [],
    # 'Plant_Matter': [],

    'Cyclopoid': [],
    'Calanoid': [],
    # 'Harpacticoid': [],
    'Nauplius_Copepod': [],
    'Bosmina': [],
    'Daphnia': [],
    'Holopedium': [],
    'Chydorus': [],
    'Diaphonosoma': [],

    'Trichocerca': [],
    'Asplanchna': [],
    'Conochilus': [],
    'Kellicottia': [],

    # 'Unknown' leaves: sample is confidently identified at the parent's level
    # (e.g. annotator knows it's a Rotifer) but not resolved to one of the
    # parent's finer sub-classes. Sit alongside the fine-grained siblings so
    # they train/predict as ordinary leaves of the tree.
    'Zoop_Unknown': [],
    'Copepoda_Unknown': [],
    'Cladocera_Unknown': [],
    'Rotifer_Unknown': [],
}

HIERNET_CLASSES = [
    'Cyclopoid', 'Calanoid',
    # 'Harpacticoid',
    'Nauplius_Copepod',
    'Bosmina', 'Daphnia', 'Holopedium', 'Chydorus', 'Diaphonosoma',
    'Trichocerca', 'Asplanchna', 'Conochilus', 'Kellicottia',
    # 'Bubbles', 'Exoskeleton', 'Fiber', 'Plant_Matter',
    'Zoop-No',
    'Zoop_Unknown', 'Copepoda_Unknown', 'Cladocera_Unknown', 'Rotifer_Unknown',
]

# 'Bosmina'/'Fiber' pool multiple on-disk folders into one leaf, matching
# config.yml's own level-0 groups exactly. 'Sididae' is an alternate/legacy
# on-disk folder name for Diaphonosoma specimens (same fix as MNR_CLASS_FOLDER_MAP,
# independent of which hierarchy is in use).
HIERNET_CLASS_FOLDER_MAP = {
    'Bosmina': ['Bosminidae', 'Bosmina_1', 'Eubosmina', 'Bosmina'],
    # 'Fiber': ['Fiber_Squiggly', 'Fiber_Hairlike', 'Fiber'],
    'Diaphonosoma': ['Diaphonosoma', 'Sididae'],
    # 'Bubbles': ['Bubbles', 'Bubble'],
    # Pools every former Zoop-No sub-class's on-disk folder into one leaf, so
    # those images still train/predict as 'Zoop-No' without their own class.
    # 'Bubble'/'Detritus'/'Small_Organism_*' are the folder names produced by the
    # Quetico_training_data batch (singular 'Bubble' as the CSVs spell it).
    'Zoop-No': ['Bubbles', 'Exoskeleton', 'Fiber_Squiggly', 'Fiber_Hairlike', 'Plant_Matter', 'Debris',
                'Bubble', 'Detritus', 'Small_Organism_Single', 'Small_Organism_Multiple'],
    # Generic-bucket folders: annotator identified the coarse class but not the
    # finer sub-class. 'Zoop_Unknown' pools the root-level 'Zooplankton' folder
    # (8 images) with Emily_test_data's held-out eval set's 'Zooplankton' folder
    # (78 images) -- read-only, and scoped to that one Emily_test_data folder via
    # the relative sub-path so no other Emily_test_data folder is pulled into
    # training.
    'Zoop_Unknown': ['Zooplankton', 'Emily_test_data/Zooplankton'],
    'Rotifer_Unknown': ['Rotifer'],
    'Copepoda_Unknown': ['Copepoda'],
    'Cladocera_Unknown': ['Cladocera'],
}

# ---------------------------------------------------------------------------
# Plankton-WHOI hierarchy — phylogenetic lineage variant
# 27 nodes | 5 levels | 14 leaf classes
# ---------------------------------------------------------------------------
whoi_adjacency_graph_l = {
      'root': ['Bacilllariophytina', 'Coscinodiscophyceae'],

      'Bacilllariophytina': ['Bacilliorhycaea', 'Mediophycaea'],
      'Coscinodiscophyceae': ['Corethron', 'Rhizosoleniaceae'],

      'Bacilliorhycaea': ['Bacillariaceae', 'Thalassionema'],
      'Mediophycaea': ['Chaetocerothophycidae', 'Thalassiosirophycidae'],

      'Chaetocerothophycidae': ['Chaetocerotales', 'Hemiaulaceae'],
      'Thalassiosirophycidae': ['Ditylum', 'Thalassiosoreles'],

      'Chaetocerotales': ['Chaetoceros', 'Leptocylindrus'],
      'Hemiaulaceae': ['Cerataulina', 'Eucampia'],
      'Thalassiosoreles': ['Skeletonema', 'Thalassiosira'],

      'Bacillariaceae': ['Cylindrotheca', 'Pseudonitzschia'],
      'Rhizosoleniaceae': ['Dactyliosolen', 'Guinardia', 'Rhizosolenia'],

      'Cylindrotheca': [],
      'Pseudonitzschia': [],
      'Thalassionema': [],
      'Chaetoceros': [],
      'Leptocylindrus': [],
      'Cerataulina': [],
      'Eucampia': [],
      'Ditylum': [],
      'Skeletonema': [],
      'Thalassiosira': [],
      'Corethron': [],
      'Dactyliosolen': [],
      'Guinardia': [],
      'Rhizosolenia': [],
  }

# ---------------------------------------------------------------------------
# Plankton-WHOI hierarchy — morphological traits variant
# 21 nodes | 3 levels | 16 leaf classes
# ---------------------------------------------------------------------------
whoi_adjacency_graph_s = {
      'root': ['Colonial', 'Unicellular'],

      'Colonial': ['C-Spines', 'C-NoSpines'],
      'Unicellular': ['U-Spines', 'U-NoSpines'],

      'C-Spines': ['Chaetoceros', 'Lauderia', 'Asterionellopsis'],
      'C-NoSpines': ['Pseudonitzschia', 'Leptocylindrus', 'Eucampia', 'Skeletonema',
                     'Dactyliosolen', 'Thalassiosira', 'Guinardia', 'Cerataulina'],
      'U-Spines': ['Corethron', 'Ditylum'],
      'U-NoSpines': ['Cylindrotheca', 'Coscinodiscus', 'Ephemera'],

      'Chaetoceros': [],
      'Lauderia': [],
      'Asterionellopsis': [],
      'Pseudonitzschia': [],
      'Leptocylindrus': [],
      'Eucampia': [],
      'Skeletonema': [],
      'Dactyliosolen': [],
      'Thalassiosira': [],
      'Guinardia': [],
      'Cerataulina': [],
      'Corethron': [],
      'Ditylum': [],
      'Cylindrotheca': [],
      'Coscinodiscus': [],
      'Ephemera': [],
  }