# Files Not Recommended For Code Submission

Do not delete these files without explicit confirmation. This list only describes what should normally be excluded from a graduation code submission package.

## Large Runtime Assets

- `data/federated_split/`: BraTS runtime data. It is too large for a code submission package.
- `data/checkpoints/sam3.pt`: model checkpoint, about several GB.
- `*.pth`, `*.pt`, `*.ckpt`: model checkpoints and training states.
- `*.nii`, `*.nii.gz`: medical image volumes.

## Generated Or Local-Only Outputs

- `logs/`: training logs and tensorboard outputs.
- `results/regenerated_figures/`: generated thesis figures.
- `generation_scripts/manifest.json` inside output folders: keep with result archives, not necessarily with source code.
- `__pycache__/`, `.pytest_cache/`, temporary notebooks, local debug files.

## What To Submit

- Source code required for figure reproduction.
- Configuration templates.
- Sample CSV templates.
- README and provenance documentation.
- A short note explaining where runtime assets should be placed on AutoDL.
