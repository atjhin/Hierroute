# Example notebooks

A complete `hierroute` workflow on the lab's zooplankton data, one notebook per stage. Run them in
order: each one uses what the previous one wrote.

| Notebook | What it does | Example data | Time (GPU) |
|---|---|---|---|
| [`preprocess.ipynb`](preprocess.ipynb) | Raw FlowCam export (mosaics + CSV) → one `.tif` per particle, in one folder per class | Lake Erie test set (53,517 particles) | < 1 min |
| [`train.ipynb`](train.ipynb) | Train a small demo model from class folders, plot its training curves and test accuracy | Training class folders at the top of `/data/Hier-Zooplankton` (300 images per class, 3 epochs) | ~1 min |
| [`inference.ipynb`](inference.ipynb) | Grade the released model on the Erie crops, classify an unclassified export, compare your own model, predict from Python | Erie crops; Emily's unclassified export (21,362 particles, 7 lakes) | ~1 min |

## Setup

```bash
pip install hierroute jupyter
jupyter lab            # then open the notebooks in this folder
```

- **Data:** the notebooks read the internal dataset at `/data/Hier-Zooplankton`. If your copy is
  elsewhere, change `DATA_DIR` in the first code cell of each notebook.
- **Outputs:** everything they write goes under `~/hierroute_example/` (`WORK_DIR`):
  - the Erie crops, about 360 MB
  - the demo model, about 115 MB
  - the results
- **Weights:** the released model weights download on first use into `~/.cache/hierroute` (115 MB per
  model).
- **No GPU?** Everything also runs on CPU, just more slowly. Training is the slowest part.

Each notebook ends with the equivalent `hierroute` terminal commands.
