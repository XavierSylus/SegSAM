# SegSAM

SegSAM is a graduation-design research codebase for federated medical image segmentation and thesis figure reproduction. The project focuses on brain tumor segmentation, foundation-model adaptation, reproducible experiment records, and clean code packaging for academic review.

This upload-ready version is prepared for GitHub display and school code submission. It contains source code, configuration files, documentation, and figure-generation utilities. It intentionally does not include medical image volumes, model checkpoints, generated logs, or private experiment archives.

## Project Motivation

Medical image segmentation projects are difficult to review if the repository mixes source code, raw data, private checkpoints, and one-off experiment files. SegSAM separates these concerns:

- Source code shows the modeling, training, aggregation, metric, and visualization logic.
- Configuration files define experimental settings without hard-coded hyperparameters.
- Reproduction scripts regenerate thesis figures from saved histories and existing prediction outputs.
- Large runtime assets are kept outside the code repository because of size, licensing, and privacy constraints.

The goal is to make the graduation project easy to inspect: a reviewer can read the architecture, verify the reproduction workflow, and understand which external assets are required for full training or figure regeneration.

## What This Project Demonstrates

SegSAM is organized to demonstrate both research ability and engineering discipline.

- Problem formulation: medical image segmentation under a federated-learning setting, with attention to heterogeneous client data and reproducible experiment records.
- Model engineering: integration of SAM-style visual foundation model components, adapter-based parameter-efficient tuning, segmentation heads, and modality-aware feature handling.
- Federated training design: client/server abstractions, aggregation utilities, local training loops, metric collection, checkpoint management, and validation hooks.
- Reproducibility: configuration-driven experiments, random seed recording, runtime metadata, Git commit tracking, and manifest generation for regenerated figures.
- Research communication: scripts export thesis figures as PNG, SVG, and PDF, with explicit provenance notes and a clear boundary between code and runtime assets.

## Repository Layout

```text
configs/
  figure_generation_config.json          Figure-generation configuration
  segmentation_samples.template.csv      Template for selecting displayed cases
  segsam_image_only.yaml                 Optional training configuration
  segsam_autodl.yaml                     AutoDL-oriented training configuration
scripts/
  generate_figures_from_existing_outputs.py
  serial_training_utils.py
  setup_serial_clients.py
src/
  agent/                                 Optional SAM3 agent integration helpers
  models/                                Adapter, freeze, and text-fusion modules
  client.py                              Client-side training logic
  federated_trainer.py                   Federated training orchestration
  integrated_model.py                    SAM3 integration and segmentation facade
  metrics.py / robust_metrics.py         Segmentation metrics
  server.py / server_manager.py          Server-side aggregation utilities
data/
  heterogeneous_dataset_loader.py        Dataset loading code only; no raw data
main.py                                  Optional training entrypoint
RESULTS_PROVENANCE.md                    Reproduction and provenance notes
SUBMISSION_EXCLUDE_LIST.md               Files intentionally excluded from submission
PACKAGE_MANIFEST.json                    Source package manifest
```

## Data Flow

```text
External BraTS data and SAM3 weights
        |
        v
Dataset loader and client setup
        |
        v
SegSAM model, adapters, training losses, metrics
        |
        v
Federated trainer, validation, checkpoints, metadata
        |
        v
Saved histories and existing predictions
        |
        v
Thesis figure generation: PNG, SVG, PDF, manifest
```

For thesis figures, the recommended workflow uses saved training history and existing prediction outputs. It does not retrain the model and does not rerun inference.

## Upload Boundary

Included in this repository:

- Python source code
- Configuration files
- CSV templates
- README and reproducibility notes
- Figure-generation scripts
- Submission manifest and exclusion list

Excluded from this repository:

- `data/federated_split/`: runtime BraTS volumes and split data
- `data/checkpoints/`: local model checkpoints such as `sam3.pt`
- `*.nii`, `*.nii.gz`: medical imaging volumes
- `*.npy`, `*.npz`: generated arrays or intermediate tensors
- `*.pt`, `*.pth`, `*.ckpt`: model weights and training checkpoints
- `logs/`, `runs/`, `wandb/`, `outputs/`, `results/`: local experiment artifacts
- `.env` and local credential files

This boundary is intentional. Raw medical data and pretrained weights may have separate licenses, access rules, privacy requirements, and file-size constraints. GitHub also blocks ordinary repository files larger than 100 MiB, so model weights and medical volumes should not be committed to the source repository.

## Environment

Recommended Python version: 3.10 or later.

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Full training or inference requires authorized third-party assets that are not bundled in this upload-ready package, including SAM3-related source code and checkpoints. Place those assets according to the paths documented in the configuration files before running the optional training path.

## Regenerate Thesis Figures

Prepare a CSV table mapping each displayed case to an existing FLAIR image, ground-truth mask, and predicted mask:

```bash
cp configs/segmentation_samples.template.csv segmentation_samples.csv
```

Example CSV row:

```csv
case_id,flair,gt,pred,slice_index
BraTS20_Training_074,/abs/path/flair_0000.png,/abs/path/true_mask_0000.png,/abs/path/pred_mask_0000.png,
```

Run figure generation from saved outputs:

```bash
python scripts/generate_figures_from_existing_outputs.py \
  --config configs/figure_generation_config.json \
  --history /abs/path/to/training_history.json \
  --segmentation-table /abs/path/to/segmentation_samples.csv \
  --output-dir /abs/path/to/segsam_regenerated_figures
```

Expected outputs:

```text
loss_curve.png / loss_curve.svg / loss_curve.pdf
metrics_dice_iou.png / metrics_dice_iou.svg / metrics_dice_iou.pdf
metrics_hd95.png / metrics_hd95.svg / metrics_hd95.pdf
segmentation_results_grid.png / segmentation_results_grid.svg / segmentation_results_grid.pdf
generation_scripts/generate_figures_from_existing_outputs.py
generation_scripts/figure_generation_config.json
generation_scripts/manifest.json
```

Figure-format note:

- Curve SVG files are pure vector graphics.
- The segmentation-grid SVG embeds bitmap image panels, so it is an SVG bitmap container.

## Optional Training Entry

The main public workflow is figure reproduction from saved outputs. The training entrypoint is retained for environments that have the required datasets, third-party SAM3 code, and checkpoints:

```bash
python main.py --config configs/segsam_image_only.yaml --rounds 1
```

The configuration files record batch size, learning rate, communication rounds, random seed, validation interval, checkpoint policy, and runtime options.

## Reproducibility Records

The project records or expects the following reproducibility fields:

- Random seed
- Runtime environment information
- Git commit hash when the code is inside a Git repository
- Training history or metric CSV/JSON
- Figure-generation command and package versions
- Manifest of generated figures and copied scripts/configuration

See `RESULTS_PROVENANCE.md` for the recommended archive structure.

## GitHub Upload Checklist

Before pushing to GitHub or submitting to school, run:

```bash
git init
git add --dry-run .
git status --short
```

Confirm that the staged file list does not include data volumes, checkpoints, generated arrays, logs, credentials, or private experiment folders. Then commit only the clean source package:

```bash
git add .
git commit -m "Prepare SegSAM graduation code release"
```

## License

This repository uses the MIT License for the original SegSAM source code included here. The license does not cover external datasets, pretrained checkpoints, SAM3 source code, SAM-Adapter, CreamFL, FedFMS, or other third-party components. Those assets must be obtained and used according to their own licenses and access rules.
