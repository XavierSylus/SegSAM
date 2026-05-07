#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "figure_generation_config.json"


@dataclass
class HistorySeries:
    rounds: list[int]
    losses: list[float]
    val_rounds: list[int]
    dice: list[float]
    iou: list[float]
    hd95_rounds: list[int]
    hd95: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SegSAM paper figures from existing outputs without running inference."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--history", type=Path, help="Override history CSV/JSON path from config.")
    parser.add_argument("--segmentation-table", type=Path, help="Override segmentation sample table path from config.")
    parser.add_argument("--output-dir", type=Path, help="Override output directory from config.")
    return parser.parse_args()


def resolve_path(value: str | Path | None, base: Path = ROOT) -> Path | None:
    if value is None or str(value) == "":
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.history is not None:
        config.setdefault("history", {})["path"] = str(args.history)
    if args.segmentation_table is not None:
        config.setdefault("segmentation_grid", {})["sample_table"] = str(args.segmentation_table)
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)


def parse_float(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text == "":
        return float("nan")
    lowered = text.lower()
    if lowered in {"nan", "none", "null"}:
        return float("nan")
    if lowered in {"inf", "+inf", "infinity", "+infinity"}:
        return float("inf")
    if lowered in {"-inf", "-infinity"}:
        return float("-inf")
    return float(text)


def finite_or_nan(value: Any) -> float:
    result = parse_float(value)
    return result if math.isfinite(result) else float("nan")


def clean_label_candidates(config: dict[str, Any], metric: str) -> list[str]:
    columns = config.get("csv_columns", {})
    candidates = columns.get(metric, [])
    if isinstance(candidates, str):
        return [candidates]
    return list(candidates)


def normalize_name(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def find_column(fieldnames: list[str], candidates: list[str], label: str) -> str:
    normalized = {normalize_name(name): name for name in fieldnames}
    for candidate in candidates:
        key = normalize_name(candidate)
        if key in normalized:
            return normalized[key]
    raise ValueError(f"Cannot find CSV column for {label}. Candidates: {candidates}. Columns: {fieldnames}")


def load_json_history(path: Path, history_cfg: dict[str, Any]) -> HistorySeries:
    text = path.read_text(encoding="utf-8-sig")
    history = json.loads(text)

    rounds = [int(v) for v in history.get("rounds", [])]
    loss_key = history_cfg.get("json_loss_key") or "avg_seg_losses"
    losses = history.get(loss_key)
    if losses is None:
        losses = history.get("avg_losses", [])
    losses = [finite_or_nan(v) for v in losses]

    val_metrics = history.get("val_metrics", [])
    val_rounds: list[int] = []
    dice: list[float] = []
    iou: list[float] = []
    hd95_rounds: list[int] = []
    hd95: list[float] = []

    for row in val_metrics:
        round_id = int(row.get("round"))
        val_rounds.append(round_id)
        dice.append(finite_or_nan(row.get("dice")))
        iou.append(finite_or_nan(row.get("iou")))
        hd95_value = finite_or_nan(row.get("hd95"))
        if math.isfinite(hd95_value):
            hd95_rounds.append(round_id)
            hd95.append(hd95_value)

    return HistorySeries(rounds, losses[: len(rounds)], val_rounds, dice, iou, hd95_rounds, hd95)


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        rows = list(reader)
        return rows, list(reader.fieldnames)


def load_wide_csv_history(path: Path, history_cfg: dict[str, Any]) -> HistorySeries:
    rows, fieldnames = read_csv_rows(path)
    round_col = find_column(fieldnames, clean_label_candidates(history_cfg, "round"), "round")
    loss_col = find_column(fieldnames, clean_label_candidates(history_cfg, "loss"), "loss")
    dice_col = find_column(fieldnames, clean_label_candidates(history_cfg, "dice"), "dice")
    iou_col = find_column(fieldnames, clean_label_candidates(history_cfg, "iou"), "iou")
    hd95_col = find_column(fieldnames, clean_label_candidates(history_cfg, "hd95"), "hd95")

    rounds: list[int] = []
    losses: list[float] = []
    val_rounds: list[int] = []
    dice: list[float] = []
    iou: list[float] = []
    hd95_rounds: list[int] = []
    hd95: list[float] = []

    for row in rows:
        round_id = int(parse_float(row[round_col]))
        rounds.append(round_id)
        losses.append(finite_or_nan(row[loss_col]))
        val_rounds.append(round_id)
        dice.append(finite_or_nan(row[dice_col]))
        iou.append(finite_or_nan(row[iou_col]))
        hd95_value = finite_or_nan(row[hd95_col])
        if math.isfinite(hd95_value):
            hd95_rounds.append(round_id)
            hd95.append(hd95_value)

    return HistorySeries(rounds, losses, val_rounds, dice, iou, hd95_rounds, hd95)


def match_long_tag(tags: set[str], candidates: list[str], label: str) -> str:
    normalized = {normalize_name(tag): tag for tag in tags}
    for candidate in candidates:
        key = normalize_name(candidate)
        if key in normalized:
            return normalized[key]
    raise ValueError(f"Cannot find long CSV tag for {label}. Candidates: {candidates}. Tags: {sorted(tags)}")


def load_long_csv_history(path: Path, history_cfg: dict[str, Any]) -> HistorySeries:
    rows, fieldnames = read_csv_rows(path)
    long_cfg = history_cfg.get("long_columns", {})
    round_col = find_column(fieldnames, [long_cfg.get("round", "step"), "round", "step"], "long round")
    tag_col = find_column(fieldnames, [long_cfg.get("tag", "tag"), "metric", "name"], "long tag")
    value_col = find_column(fieldnames, [long_cfg.get("value", "value"), "scalar"], "long value")
    all_tags = {row[tag_col] for row in rows}
    tag_cfg = history_cfg.get("long_tags", {})

    selected = {
        "loss": match_long_tag(all_tags, tag_cfg.get("loss", clean_label_candidates(history_cfg, "loss")), "loss"),
        "dice": match_long_tag(all_tags, tag_cfg.get("dice", clean_label_candidates(history_cfg, "dice")), "dice"),
        "iou": match_long_tag(all_tags, tag_cfg.get("iou", clean_label_candidates(history_cfg, "iou")), "iou"),
        "hd95": match_long_tag(all_tags, tag_cfg.get("hd95", clean_label_candidates(history_cfg, "hd95")), "hd95"),
    }

    values_by_metric: dict[str, dict[int, float]] = {key: {} for key in selected}
    for row in rows:
        for metric, tag in selected.items():
            if row[tag_col] == tag:
                values_by_metric[metric][int(parse_float(row[round_col]))] = finite_or_nan(row[value_col])

    rounds = sorted(values_by_metric["loss"])
    val_rounds = sorted(set(values_by_metric["dice"]) & set(values_by_metric["iou"]))
    hd95_rounds = [
        round_id
        for round_id in sorted(values_by_metric["hd95"])
        if math.isfinite(values_by_metric["hd95"][round_id])
    ]

    return HistorySeries(
        rounds=rounds,
        losses=[values_by_metric["loss"][round_id] for round_id in rounds],
        val_rounds=val_rounds,
        dice=[values_by_metric["dice"][round_id] for round_id in val_rounds],
        iou=[values_by_metric["iou"][round_id] for round_id in val_rounds],
        hd95_rounds=hd95_rounds,
        hd95=[values_by_metric["hd95"][round_id] for round_id in hd95_rounds],
    )


def load_history(path: Path, history_cfg: dict[str, Any]) -> HistorySeries:
    if not path.exists():
        raise FileNotFoundError(f"Missing history file: {path}")
    suffix = path.suffix.lower()
    if suffix == ".json":
        return load_json_history(path, history_cfg)
    if suffix == ".csv":
        schema = history_cfg.get("csv_schema", "auto").lower()
        if schema == "wide":
            return load_wide_csv_history(path, history_cfg)
        if schema == "long":
            return load_long_csv_history(path, history_cfg)
        rows, fieldnames = read_csv_rows(path)
        lower_fields = {normalize_name(name) for name in fieldnames}
        if {"tag", "value"} <= lower_fields or {"metric", "value"} <= lower_fields:
            return load_long_csv_history(path, history_cfg)
        return load_wide_csv_history(path, history_cfg)
    raise ValueError(f"Unsupported history file type: {path}")


def setup_matplotlib(config: dict[str, Any]) -> None:
    rc = config.get("matplotlib_rc", {})
    if "font.family" not in rc:
        rc["font.family"] = "DejaVu Sans"
    if "pdf.fonttype" not in rc:
        rc["pdf.fonttype"] = 42
    if "ps.fonttype" not in rc:
        rc["ps.fonttype"] = 42
    if "svg.fonttype" not in rc:
        rc["svg.fonttype"] = "none"
    plt.rcParams.update(rc)


def save_figure(fig: plt.Figure, out_dir: Path, basename: str, output_cfg: dict[str, Any]) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(output_cfg.get("dpi", 300))
    formats = output_cfg.get("formats", ["png", "svg", "pdf"])
    paths: list[Path] = []
    for ext in formats:
        ext_clean = str(ext).lower().lstrip(".")
        path = out_dir / f"{basename}.{ext_clean}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        paths.append(path)
    return paths


def require_series(values: list[Any], label: str) -> None:
    if not values:
        raise ValueError(f"No data available for {label}")


def plot_loss_curve(series: HistorySeries, config: dict[str, Any], out_dir: Path) -> list[Path]:
    require_series(series.rounds, "training rounds")
    require_series(series.losses, "training loss")
    current_round = max(series.rounds)
    style = config.get("curve_style", {})
    fig, ax = plt.subplots(figsize=tuple(style.get("loss_figsize", [12, 6])))
    fig.patch.set_facecolor("#ffffff")
    ax.plot(
        series.rounds,
        series.losses,
        color=style.get("loss_color", "#ff7f0e"),
        marker=style.get("loss_marker", "s"),
        linewidth=float(style.get("line_width", 2.0)),
        markersize=float(style.get("marker_size", 4.0)),
        label=style.get("loss_label", "Seg loss"),
    )
    ax.set_title(f"Training Loss Curves (Round {current_round})", fontweight="bold")
    ax.set_xlabel("Round")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=float(style.get("grid_alpha", 0.3)))
    ax.legend(loc="best")
    fig.tight_layout()
    paths = save_figure(fig, out_dir, "loss_curve", config.get("output", {}))
    plt.close(fig)
    return paths


def plot_dice_iou_curve(series: HistorySeries, config: dict[str, Any], out_dir: Path) -> list[Path]:
    require_series(series.val_rounds, "validation rounds")
    current_round = max(series.val_rounds)
    style = config.get("curve_style", {})
    fig, ax = plt.subplots(figsize=tuple(style.get("metric_figsize", [12, 6])))
    fig.patch.set_facecolor("#ffffff")
    ax.plot(
        series.val_rounds,
        series.dice,
        color=style.get("dice_color", "blue"),
        marker=style.get("dice_marker", "s"),
        linewidth=float(style.get("line_width", 2.0)),
        markersize=float(style.get("marker_size", 4.0)),
        label="Dice Score",
    )
    ax.plot(
        series.val_rounds,
        series.iou,
        color=style.get("iou_color", "green"),
        marker=style.get("iou_marker", "^"),
        linewidth=float(style.get("line_width", 2.0)),
        markersize=float(style.get("marker_size", 4.0)),
        label="IoU Score",
    )
    ax.set_title(f"Validation Metrics: Dice & IoU (Round {current_round})", fontweight="bold")
    ax.set_xlabel("Round")
    ax.set_ylabel("Score")
    ax.set_ylim(tuple(style.get("score_ylim", [0.0, 1.0])))
    ax.grid(True, alpha=float(style.get("grid_alpha", 0.3)))
    ax.legend(loc="best")
    fig.tight_layout()
    paths = save_figure(fig, out_dir, "metrics_dice_iou", config.get("output", {}))
    plt.close(fig)
    return paths


def plot_hd95_curve(series: HistorySeries, config: dict[str, Any], out_dir: Path) -> list[Path]:
    require_series(series.hd95_rounds, "finite HD95 rounds")
    current_round = max(series.val_rounds or series.hd95_rounds)
    style = config.get("curve_style", {})
    fig, ax = plt.subplots(figsize=tuple(style.get("metric_figsize", [12, 6])))
    fig.patch.set_facecolor("#ffffff")
    ax.plot(
        series.hd95_rounds,
        series.hd95,
        color=style.get("hd95_color", "red"),
        marker=style.get("hd95_marker", "x"),
        linewidth=float(style.get("line_width", 2.0)),
        markersize=float(style.get("marker_size", 4.0)),
        label="HD95",
    )
    ax.set_title(f"Validation Metric: HD95 (Round {current_round})", fontweight="bold")
    ax.set_xlabel("Round")
    ax.set_ylabel("HD95 (mm)")
    ax.grid(True, alpha=float(style.get("grid_alpha", 0.3)))
    ax.legend(loc="best")
    fig.tight_layout()
    paths = save_figure(fig, out_dir, "metrics_hd95", config.get("output", {}))
    plt.close(fig)
    return paths


def is_nifti_path(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def normalize_flair(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("FLAIR image contains no finite values")
    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        low, high = float(np.min(finite)), float(np.max(finite))
    if high <= low:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - low) / (high - low), 0.0, 1.0)


def normalize_mask(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return (arr > 0.5).astype(np.float32)


def load_nifti_slice(path: Path, slice_index: int) -> np.ndarray:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("nibabel is required to load .nii/.nii.gz files") from exc

    data = nib.load(str(path)).get_fdata()
    if data.ndim != 3:
        raise ValueError(f"NIfTI file must be 3D: {path}")
    if slice_index < 0 or slice_index >= data.shape[2]:
        raise IndexError(f"slice_index={slice_index} out of range for {path}, depth={data.shape[2]}")
    return data[:, :, slice_index]


def choose_slice_index(sample: dict[str, Any], grid_cfg: dict[str, Any]) -> int | None:
    if sample.get("slice_index") not in {None, ""}:
        return int(sample["slice_index"])
    policy = sample.get("slice_policy") or grid_cfg.get("slice_policy", "require_explicit")
    if policy == "require_explicit":
        return None
    if policy != "largest_gt":
        raise ValueError(f"Unknown slice_policy: {policy}")

    gt_path = resolve_path(sample.get("gt"))
    if gt_path is None or not is_nifti_path(gt_path):
        raise ValueError("slice_policy=largest_gt requires a NIfTI gt path")

    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("nibabel is required for slice_policy=largest_gt") from exc

    gt_data = nib.load(str(gt_path)).get_fdata()
    if gt_data.ndim != 3:
        raise ValueError(f"GT NIfTI file must be 3D: {gt_path}")
    per_slice = (gt_data > 0).sum(axis=(0, 1))
    return int(np.argmax(per_slice))


def load_panel(path_value: str, kind: str, sample: dict[str, Any], grid_cfg: dict[str, Any]) -> np.ndarray:
    path = resolve_path(path_value)
    if path is None:
        raise ValueError(f"Missing {kind} path for sample {sample.get('case_id', '')}")
    if not path.exists():
        raise FileNotFoundError(f"Missing {kind} file: {path}")

    if is_nifti_path(path):
        slice_index = choose_slice_index(sample, grid_cfg)
        if slice_index is None:
            raise ValueError(f"NIfTI {kind} file requires slice_index: {path}")
        array = load_nifti_slice(path, slice_index)
    else:
        with Image.open(path) as image:
            if kind == "flair":
                array = np.asarray(image.convert("L"))
            else:
                array = np.asarray(image.convert("L"))

    if kind == "flair":
        return normalize_flair(array)
    return normalize_mask(array)


def collect_segmentation_samples(grid_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    table_path = resolve_path(grid_cfg.get("sample_table"))
    if table_path is not None:
        if not table_path.exists():
            raise FileNotFoundError(f"Missing segmentation sample table: {table_path}")
        with table_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"Segmentation sample table has no header: {table_path}")
            for row in reader:
                if any(str(row.get(key, "")).strip() for key in ("flair", "gt", "pred")):
                    samples.append(dict(row))

    for sample in grid_cfg.get("samples", []):
        samples.append(dict(sample))

    if not samples:
        raise ValueError("segmentation_grid is enabled, but no samples were provided")
    return samples


def plot_segmentation_grid(config: dict[str, Any], out_dir: Path) -> list[Path]:
    grid_cfg = config.get("segmentation_grid", {})
    samples = collect_segmentation_samples(grid_cfg)
    n_cols = int(grid_cfg.get("n_cols", 4))
    n_rows = int(math.ceil(len(samples) / n_cols))
    figsize = grid_cfg.get("figsize")
    if figsize is None:
        figsize = [n_cols * 3.1, n_rows * 7.8]

    fig, axes = plt.subplots(n_rows * 3, n_cols, figsize=tuple(figsize), dpi=int(config.get("output", {}).get("dpi", 300)))
    fig.patch.set_facecolor("#ffffff")
    axes_array = np.asarray(axes).reshape(n_rows * 3, n_cols)

    row_titles = grid_cfg.get("row_titles", ["FLAIR", "GT (WT)", "Pred (WT)"])
    for index, sample in enumerate(samples):
        row_block = (index // n_cols) * 3
        col = index % n_cols
        case_id = sample.get("case_id") or Path(str(sample.get("flair", ""))).parent.name

        flair = load_panel(str(sample.get("flair", "")), "flair", sample, grid_cfg)
        gt = load_panel(str(sample.get("gt", "")), "gt", sample, grid_cfg)
        pred = load_panel(str(sample.get("pred", "")), "pred", sample, grid_cfg)
        panels = [(flair, "gray"), (gt, "Reds"), (pred, "Blues")]

        for row_offset, (array, cmap) in enumerate(panels):
            ax = axes_array[row_block + row_offset, col]
            ax.imshow(array, cmap=cmap, vmin=0, vmax=1)
            ax.axis("off")
            if row_offset == 0:
                ax.set_title(f"{case_id}\n{row_titles[row_offset]}", fontsize=7, pad=2)
            else:
                ax.set_title(row_titles[row_offset], fontsize=8, pad=2)

    for index in range(len(samples), n_rows * n_cols):
        row_block = (index // n_cols) * 3
        col = index % n_cols
        for row_offset in range(3):
            axes_array[row_block + row_offset, col].axis("off")

    fig.suptitle(
        grid_cfg.get("title", "Segmentation Results: FLAIR | GT (WT) | Pred (WT)"),
        fontsize=13,
        fontweight="bold",
        y=float(grid_cfg.get("title_y", 1.002)),
    )
    fig.tight_layout()
    paths = save_figure(fig, out_dir, "segmentation_results_grid", config.get("output", {}))
    plt.close(fig)
    return paths


def get_git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip()


def package_versions() -> dict[str, str]:
    versions = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "pillow": Image.__version__,
    }
    try:
        import nibabel as nib

        versions["nibabel"] = nib.__version__
    except Exception:
        versions["nibabel"] = "not imported"
    return versions


def copy_generation_files(config_path: Path, out_dir: Path, config: dict[str, Any]) -> list[Path]:
    provenance_cfg = config.get("provenance", {})
    generation_dir = out_dir / provenance_cfg.get("generation_dir", "generation_scripts")
    generation_dir.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    if provenance_cfg.get("copy_script", True):
        script_src = Path(__file__).resolve()
        script_dst = generation_dir / script_src.name
        if script_src != script_dst.resolve():
            shutil.copy2(script_src, script_dst)
        copied.append(script_dst)

    if provenance_cfg.get("copy_config", True):
        config_dst = generation_dir / config_path.name
        if config_path.resolve() != config_dst.resolve():
            shutil.copy2(config_path, config_dst)
        copied.append(config_dst)

    return copied


def write_manifest(
    out_dir: Path,
    config_path: Path,
    history_path: Path,
    generated: list[Path],
    copied: list[Path],
    config: dict[str, Any],
) -> Path:
    generation_dir = out_dir / config.get("provenance", {}).get("generation_dir", "generation_scripts")
    manifest_path = generation_dir / "manifest.json"
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "git_commit": get_git_commit(),
        "config_path": str(config_path),
        "history_path": str(history_path),
        "generated_files": [str(path) for path in generated],
        "copied_generation_files": [str(path) for path in copied],
        "package_versions": package_versions(),
        "svg_notes": {
            "loss_curve": "pure vector",
            "metrics_dice_iou": "pure vector",
            "metrics_hd95": "pure vector",
            "segmentation_results_grid": "bitmap container when image panels are embedded",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    apply_cli_overrides(config, args)
    setup_matplotlib(config)

    out_dir = resolve_path(config.get("output_dir"))
    if out_dir is None:
        raise ValueError("output_dir is required in config")
    history_path = resolve_path(config.get("history", {}).get("path"))
    if history_path is None:
        raise ValueError("history.path is required in config")

    series = load_history(history_path, config.get("history", {}))
    generated: list[Path] = []
    generated.extend(plot_loss_curve(series, config, out_dir))
    generated.extend(plot_dice_iou_curve(series, config, out_dir))
    generated.extend(plot_hd95_curve(series, config, out_dir))

    if config.get("segmentation_grid", {}).get("enabled", True):
        generated.extend(plot_segmentation_grid(config, out_dir))

    copied = copy_generation_files(config_path, out_dir, config)
    manifest_path = write_manifest(out_dir, config_path, history_path, generated, copied, config)

    print("Generated files:")
    for path in generated:
        print(path)
    print("Generation files:")
    for path in copied:
        print(path)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

