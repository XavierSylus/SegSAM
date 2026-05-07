"""SegSAM optional training entrypoint.

The primary thesis workflow regenerates figures from saved outputs via
`scripts/generate_figures_from_existing_outputs.py`. This entrypoint is retained
for compatibility when an environment needs to inspect or rerun the image-only
segmentation training pipeline.
"""

import argparse
import sys
from pathlib import Path


project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config_manager import FederatedConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SegSAM optional image segmentation training entrypoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Main thesis figures should be regenerated from saved outputs:\n"
            "  # python scripts/generate_figures_from_existing_outputs.py --help\n\n"
            "  # Optional smoke run for the retained training path:\n"
            "  python main.py --config configs/segsam_image_only.yaml --rounds 1\n"
            "  python main.py --help\n"
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config file",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Override the data root from the config file",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Override the number of training rounds",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override the batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override the learning rate",
    )
    parser.add_argument(
        "--lambda_cream",
        type=float,
        default=None,
        help="Compatibility override for the auxiliary loss weight",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the random seed",
    )
    parser.add_argument(
        "--use_mock",
        action="store_true",
        help="Use the mock SAM3 model instead of real checkpoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override the training device",
    )
    return parser


def apply_cli_overrides(config: FederatedConfig, args: argparse.Namespace) -> None:
    overrides_applied = []

    if args.data_root is not None:
        config.data_root = args.data_root
        overrides_applied.append(f"data_root={args.data_root}")
    if args.rounds is not None:
        config.rounds = args.rounds
        overrides_applied.append(f"rounds={args.rounds}")
    if args.batch_size is not None:
        config.batch_size = args.batch_size
        overrides_applied.append(f"batch_size={args.batch_size}")
    if args.lr is not None:
        config.lr = args.lr
        overrides_applied.append(f"lr={args.lr}")
    if args.lambda_cream is not None:
        config.lambda_cream = args.lambda_cream
        overrides_applied.append(f"lambda_cream={args.lambda_cream}")
    if args.seed is not None:
        config.seed = args.seed
        overrides_applied.append(f"seed={args.seed}")
    if args.use_mock:
        config.use_mock = True
        overrides_applied.append("use_mock=True")
    if args.device is not None:
        config.device = args.device
        overrides_applied.append(f"device={args.device}")

    if config.log_dir is None:
        config.log_dir = str(Path(config.data_root) / "logs")
        overrides_applied.append(f"log_dir={config.log_dir} (default)")

    if overrides_applied:
        print(f"[OK] Applied CLI/default overrides: {', '.join(overrides_applied)}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    print("=" * 60)
    print("Loading SegSAM training configuration...")
    print("=" * 60)
    try:
        config = FederatedConfig.from_yaml(args.config)
        print(f"[OK] Loaded config: {args.config}")
        print(f"  Summary: {config}")
    except Exception as exc:
        print(f"[ERROR] Failed to load config: {exc}")
        import traceback

        traceback.print_exc()
        return 1

    apply_cli_overrides(config, args)

    try:
        from src.federated_trainer import FederatedTrainer
    except Exception as exc:
        print(f"[ERROR] Failed to import training stack: {exc}")
        import traceback

        traceback.print_exc()
        return 1

    try:
        trainer = FederatedTrainer(config)
        return trainer.train()
    except Exception as exc:
        print(f"Training failed: {exc}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
