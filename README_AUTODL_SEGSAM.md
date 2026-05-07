# SegSAM AutoDL Notes

This note is kept for running SegSAM on AutoDL. Use `README.md` as the authoritative project guide.

The current SegSAM workflow regenerates thesis figures from saved training history and existing segmentation outputs. It does not require retraining or rerunning inference.

Recommended command:

```bash
python scripts/generate_figures_from_existing_outputs.py \
  --config configs/figure_generation_config.json \
  --history /abs/path/to/training_history.json \
  --segmentation-table /abs/path/to/segmentation_samples.csv \
  --output-dir /abs/path/to/segsam_regenerated_figures
```
