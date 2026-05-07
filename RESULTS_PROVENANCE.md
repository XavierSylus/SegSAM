# Results Provenance

This project regenerates thesis figures from saved experiment outputs. The figure generation workflow does not retrain the segmentation model and does not rerun inference.

## Required Inputs

- Saved training history: `training_history.json` or a CSV with round, loss, Dice, IoU, and HD95 columns.
- Existing visual outputs: FLAIR image panels, ground-truth WT masks, and predicted WT masks.
- Segmentation sample table: a CSV mapping each displayed case to its FLAIR, GT, and prediction file.

## Reproducibility Fields To Preserve

When available, keep these files together with the generated figures:

- `training_history.json`
- `run_metadata.json`
- `segmentation_samples.csv`
- generated `generation_scripts/manifest.json`
- the exact figure generation config used for the thesis

The manifest written by the script records the command, input paths, package versions, timestamp, and generated files.

## Figure Format Notes

- Loss, Dice/IoU, and HD95 curves are regenerated as vector Matplotlib figures and exported to PNG, SVG, and PDF.
- The segmentation result grid contains image panels. Its SVG output is a bitmap container, not a pure vector drawing.

## Submission Boundary

Large runtime assets should not be included in the code submission. Keep datasets, model weights, logs, and generated figures in the experiment archive or AutoDL storage, and submit only code, configuration, templates, and documentation unless the school explicitly asks for runtime assets.
