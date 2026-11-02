"""Train and evaluate the IMD rainfall, Tmax, and Tmin one-day baselines."""

from __future__ import annotations

import argparse
import json

from climate_forecast import train_variable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variables",
        nargs="+",
        choices=("rainfall", "tmax", "tmin"),
        default=["rainfall", "tmax", "tmin"],
        help="Climate variables to train (default: all available variables).",
    )
    parser.add_argument(
        "--max-samples-per-day",
        type=int,
        default=1500,
        help="Maximum deterministic training samples selected per day and variable.",
    )
    args = parser.parse_args()
    if args.max_samples_per_day < 1:
        parser.error("--max-samples-per-day must be positive.")

    for variable in args.variables:
        metrics = train_variable(
            variable,
            max_samples_per_day=args.max_samples_per_day,
        )
        print(
            json.dumps(
                {
                    "variable": variable,
                    "model_name": metrics["model_name"],
                    "validation": metrics["candidate_xgboost_validation_metrics"],
                    "validation_persistence": metrics["persistence_validation_metrics"],
                    "test": metrics["test_metrics"],
                    "test_persistence": metrics["persistence_test_metrics"],
                    "training_period": metrics["training_period"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
