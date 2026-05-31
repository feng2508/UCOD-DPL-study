# UCOD-DPL Study and Evaluation Workspace

This repository is a local study and reproduction workspace for **UCOD-DPL: Unsupervised Camouflaged Object Detection via Dynamic Pseudo-label Learning**. It is based on the UCOD-DPL/CORAL research code and adds practical notebooks for evaluating saved predictions, analyzing COD metrics, and visualizing failure cases.

This is not the original paper repository. The original work should be cited using the paper reference below.

## What This Workspace Contains

- UCOD-DPL and CORAL model code, configs, and launch scripts.
- Evaluation notebooks for saved prediction masks.
- Per-image metric analysis workflows.
- Failure-case visualization for poor COD predictions.
- Reading notes under `docs/`.

Large local artifacts are intentionally excluded from Git:

- `datasets/`
- `results/`
- `weights/`
- zip files and generated outputs

## Project Layout

```text
UCOD-DPL/
  configs/                 # Model and dataset configs
  data/                    # Dataset and feature utilities
  docs/                    # Paper-reading notes and study documentation
  engine/                  # Runner, config, logging, and metric code
  models/                  # Model architectures and modules
  notebooks/               # Local evaluation and analysis notebooks
  scripts/                 # Train/eval launch entrypoints
  requirement.txt          # Python dependencies from the project
```

Important notebooks:

```text
notebooks/evaluate-cod-predictions.ipynb
notebooks/ucod-evaluate-dataset.ipynb
notebooks/visualize-worst-cod-predictions.ipynb
```

## Environment

This project was used on macOS with Conda.

```zsh
conda create -n coral python=3.9 -y
conda activate coral
pip install -r requirement.txt
```

If you use Jupyter Notebook, register the Conda environment as a kernel:

```zsh
conda activate coral
python -m ipykernel install --user --name coral --display-name coral
```

## Data and Checkpoints

Expected local dataset layout:

```text
datasets/RefCOD/
  CHAMELEON/
    im/
    gt/
  TE-CAMO/
    im/
    gt/
  TE-COD10K/
    im/
    gt/
  NC4K/
    im/
    gt/
  TR-CAMO/
  TR-COD10K/
```

Expected prediction layout:

```text
results/preds/
  CHAMELEON/
  TE-CAMO/
  TE-COD10K/
  NC4K/
```

Prediction masks should use filenames matching the GT masks. The metric code resizes predictions to the GT size before computing scores.

## Evaluate Saved Predictions

To compute COD metrics from saved prediction masks without rerunning model inference:

```python
from pathlib import Path
from engine.utils.metrics.metric import calculate_cod_metrics

gt_dir = Path("datasets/RefCOD/TE-COD10K/gt")
pred_dir = Path("results/preds/TE-COD10K")

metrics = calculate_cod_metrics(str(gt_dir), str(pred_dir), verbose=True)
{k: round(float(v), 4) for k, v in metrics.items()}
```

The main metric keys are:

```text
E_MAX, E_MEAN, F_MAX, F_MEAN, SMeasure, MAE, WFM
```

For `MAE`, lower is better. For the other metrics, higher is better.

## Per-Image Analysis

The notebook `notebooks/evaluate-cod-predictions.ipynb` computes per-image metrics and saves:

```text
results/metric_analysis/te-cod10k-per-image-metrics.csv
```

The notebook `notebooks/visualize-worst-cod-predictions.ipynb` reads that CSV, sorts poor predictions, and visualizes:

- original image
- ground-truth mask
- predicted mask
- prediction overlay

It also saves visualization outputs under:

```text
results/metric_analysis/
```

## Training and Model Evaluation

First-stage UCOD-DPL training:

```zsh
bash ./scripts/launch_train_first_stage.sh -c ./configs/uscod/UCOD-DPL_dinov2.py
```

First-stage UCOD-DPL checkpoint evaluation:

```zsh
bash ./scripts/launch_val_first_stage.sh \
  -c ./configs/uscod/UCOD-DPL_dinov2.py \
  -m path/to/UCOD-DPL-dinov2/model
```

Second-stage CORAL evaluation:

```zsh
bash ./scripts/launch_val_second_stage.sh \
  -c ./configs/uscod/CORAL_dinov2.py \
  -m path/to/UCOD-DPL-dinov2/model \
  -r path/to/CORAL_dinov2/model
```

## Study Notes

Paper-reading and method-understanding notes are kept in:

```text
docs/
```

Current notes:

```text
docs/ucod-dp-reading-note.md
docs/ucod-dpl-code-map.md
```

The notes are intended to record:

- the problem the paper solves
- the main method idea
- paper concepts mapped to implementation files and functions
- evidence from tables and figures
- ablation limitations
- failure modes found through local evaluation
- ideas that may be reusable in future UCOD work

## Reference

```bibtex
@inproceedings{yan2025ucod,
  title={UCOD-DPL: Unsupervised Camouflaged Object Detection via Dynamic Pseudo-label Learning},
  author={Yan, Weiqi and Chen, Lvhai and Kou, Huaijia and Zhang, Shengchuan and Zhang, Yan and Cao, Liujuan},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={30365--30375},
  year={2025}
}
```
