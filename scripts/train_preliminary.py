"""Train the preliminary main run or its clean-only control.

uv run python scripts/train_preliminary.py DATASET RUN_DIRECTORY
"""

import argparse

from lenahs.model.architecture import CorrectionModelConfig, CorrectionRunConfig
from lenahs.training.loop import PreliminaryRunSettings, fit


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("run_directory")
    parser.add_argument("--max-epochs", type=int, required=True)
    parser.add_argument("--clean-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plateau-patience", type=int, default=3)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    settings = PreliminaryRunSettings(
        dataset_root=arguments.dataset,
        include_noisy=not arguments.clean_only,
        run=CorrectionRunConfig(
            model=CorrectionModelConfig(hidden_width=128, correction_recurrent=True),
            observed_steps=10,
            correction_steps=5,
        ),
        max_epochs=arguments.max_epochs,
        plateau_patience=arguments.plateau_patience,
        early_stopping_patience=arguments.early_stopping_patience,
        num_workers=arguments.workers,
        seed=arguments.seed,
    )
    test = fit(settings, arguments.run_directory, resume=arguments.resume)
    for name, value in test.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
