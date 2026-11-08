"""
constants.py -- Phytoplankton label map for the binary "phytoplankton vs not"
task.

The FlowCam particle-export CSVs carry 24 distinct raw values in their ``Class``
column (see the counts printed by
``phytoplankton/notebooks/pre_process_data.ipynb``). ``pre_process_data`` crops
one image per particle into ``Processed_data/<raw class>/``; the dataloader then
collapses those raw-class folders into the two model classes using the maps
below.

Every raw class is assigned exactly one bucket in ``CLASS_MAP``:

    "Phytoplankton"      real phytoplankter
    "Not Phytoplankton"  zooplankton, ciliates, bubbles, generic non-phyto
    EXCLUDE              drop from the dataset entirely -- literal duplicate
                        images (a particle that got stuck and was imaged several
                        times)
    TBD                  not decided yet. Probably phytoplankton, but not
                        verifiable at this imaging resolution. NOT USED at the
                        moment -- the dataloader ignores these rows, same as
                        EXCLUDE. Parked here so the call is visible and easy to
                        fold in later.
"""

# The two model classes, in fixed order. Index is the training label:
#   0 -> "Not Phytoplankton", 1 -> "Phytoplankton"
BINARY_CLASSES = ["Not Phytoplankton", "Phytoplankton"]

# Non-class bucket sentinels used as CLASS_MAP values.
EXCLUDE = "EXCLUDE"
TBD = "TBD"


# ---------------------------------------------------------------------------
# Single source of truth: raw CSV "Class" value -> bucket
# ---------------------------------------------------------------------------
CLASS_MAP = {
    # --- Phytoplankton -----------------------------------------------------
    "Diatom":            "Phytoplankton",
    "Dinobryon":         "Phytoplankton",
    "Dinoflagellate":    "Phytoplankton",
    "Phytoplankton":     "Phytoplankton",
    "SP_Colony":         "Phytoplankton",
    "EL_Colony":         "Phytoplankton",
    "Colony":            "Phytoplankton",

    # --- Not phytoplankton ----------------------------------------------------
    "Not Phytoplankton": "Not Phytoplankton",
    "Zooplankton":       "Not Phytoplankton",
    "Ciliate":           "Not Phytoplankton",
    "Bubble":            "Not Phytoplankton",

    # --- Excluded: duplicate imaging of a single stuck particle ------------
    "Duplicate":         EXCLUDE,
    "Duplicates":        EXCLUDE,   # accidental second spelling of "Duplicate"

    # --- TBD: likely phytoplankton, unverifiable at this resolution -------
    #     Currently ignored by the dataloader (see module docstring).
    "Unclear":           TBD,
    "Unclear_Filament":  TBD,
    "Filament_Likely":   TBD,
    "MultiClass":        TBD,
    "Blurry":            TBD,
    "Too blurry":        TBD,
    "DarkClump":         TBD,
    "Unknown":           TBD,
    "Unknown sp1":       TBD,
    "Empty Cells":       TBD,
    "Empty Cell":        TBD,   # singular variant present in some CSVs (~20 rows);
                                # not in the hand-written list but treated as
                                # "Empty Cells". Revisit with the other TBDs.
}

# Every raw class observed in the CSVs (= the keys of CLASS_MAP), sorted.
RAW_CLASSES = sorted(CLASS_MAP)


# ---------------------------------------------------------------------------
# Derived views for dataloader construction
# ---------------------------------------------------------------------------

# raw class -> "Phytoplankton" / "Not Phytoplankton"  (resolved labels only)
CLASS_TO_BINARY = {
    raw: bucket for raw, bucket in CLASS_MAP.items() if bucket in BINARY_CLASSES
}

# model class -> list of raw-class folder names to pool into it.
# Directly usable as HierNet ImageDataset's `class_folder_map` / to build
# `merge_classes` arguments.
BINARY_CLASS_FOLDERS = {
    model_class: sorted(
        raw for raw, bucket in CLASS_MAP.items() if bucket == model_class
    )
    for model_class in BINARY_CLASSES
}

# raw classes to drop from the dataset outright
EXCLUDE_CLASSES = {raw for raw, bucket in CLASS_MAP.items() if bucket == EXCLUDE}

# raw classes parked for a later decision (currently dropped, like EXCLUDE)
TBD_CLASSES = {raw for raw, bucket in CLASS_MAP.items() if bucket == TBD}

# everything the current binary dataloader should skip
IGNORED_CLASSES = EXCLUDE_CLASSES | TBD_CLASSES


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
assert set(CLASS_MAP.values()) <= set(BINARY_CLASSES) | {EXCLUDE, TBD}
assert set(CLASS_TO_BINARY) | EXCLUDE_CLASSES | TBD_CLASSES == set(CLASS_MAP)
assert sum(len(v) for v in BINARY_CLASS_FOLDERS.values()) == len(CLASS_TO_BINARY)
