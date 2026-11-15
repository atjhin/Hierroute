# Example notebooks

A complete `hierroute` workflow on the lab's zooplankton data, one notebook per stage.
- `preprocess.ipynb` shows how raw FlowCam exports become particle crops.
- `train.ipynb` and `inference.ipynb` reproduce how the released `noise` model was trained and tested
  on the Quetico messy-background test set. `inference.ipynb` gets the original evaluation's results
  exactly; a retrain lands close to the released model but not bit-for-bit (GPU training isn't fully
  deterministic).

| Notebook | What it does | Example data | Time (GPU) |
|---|---|---|---|
| [`preprocess.ipynb`](preprocess.ipynb) | Raw FlowCam export (mosaics + CSV) → one `.tif` per particle, in one folder per class | Lake Erie test set (53,517 particles) | < 1 min |
| [`train.ipynb`](train.ipynb) | Train the released `noise` model's configuration (Swin-T + CNN experts, background-noise augmentation, 40 epochs) and compare its test scores with the released model's | Class folders in `/data/Hier-Zooplankton`, `IssacData/` and `Quetico_training_data/` (71,723 images) | 20 minutes (`QUICK = True`: a few minutes) |
| [`inference.ipynb`](inference.ipynb) | Grade the released model on the Quetico messy-background test set, compare your trained model, classify the raw export, predict from Python | `quetico/Messy/` (25,564 particles, 11 runs) | ~1 min |

## Setup

```bash
pip install hierroute jupyter
jupyter lab            # then open the notebooks in this folder
```

- **Data:** the notebooks read the internal dataset at `/data/Hier-Zooplankton`. If your copy is
  elsewhere, change `DATA_DIR` in the first code cell of each notebook.
- **Outputs** (paths relative to the notebooks; git ignores all of them):
  - `train.ipynb`: the trained model (about 115 MB) in `../training_result/`
  - `inference.ipynb`: the evaluation results in `../inference_result/`
  - `preprocess.ipynb`: the Erie crops (about 360 MB) in `~/hierroute_example/` (`WORK_DIR`)
- **Weights:** the released model weights download on first use into `~/.cache/hierroute` (115 MB per
  model).
- **No GPU?** Everything also runs on CPU, just more slowly. Training is the slowest part.

Each notebook ends with the equivalent `hierroute` terminal commands.
