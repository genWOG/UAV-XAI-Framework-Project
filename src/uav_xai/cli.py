import argparse
import sys

from .core.runner import ExperimentRunner
from .core.benchmark import BenchmarkRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uav-xai",
        description=(
            "Explainable AI toolkit "
            "for autonomous UAV decision systems."
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    experiment_parser = (
        subparsers.add_parser(
            "experiment",
            help="Run an XAI experiment.",
        )
    )

    experiment_parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration.",
    )

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run a repeated XAI benchmark.",
    )

    benchmark_parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML benchmark configuration.",
    )

    return parser


def main() -> int:
    parser = build_parser()

    args = parser.parse_args()

    if args.command == "experiment":
        try:
            runner = ExperimentRunner(
                args.config
            )

            runner.run()

            return 0

        except Exception as exc:
            print(
                f"ERROR: {exc}",
                file=sys.stderr,
            )

            return 1

    if args.command == "benchmark":
        try:
            runner = BenchmarkRunner(
                args.config
            )

            runner.run()

            return 0

        except Exception as exc:
            print(
                f"ERROR: {exc}",
                file=sys.stderr,
            )

            return 1

    parser.print_help()

    return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )