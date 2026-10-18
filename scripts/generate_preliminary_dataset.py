"""Generate the preliminary salt-and-pepper dataset.

uv run python scripts/generate_preliminary_dataset.py ROOT
"""

import argparse
import os

from lenahs.component_tree.types import ComponentTreeKind
from lenahs.dataset.build import generate_dataset
from lenahs.generation.images import SquareImageConfig
from lenahs.generation.noise import SaltAndPepperNoise


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--train", type=int, default=10_000)
    parser.add_argument("--val", type=int, default=1_000)
    parser.add_argument("--test", type=int, default=1_000)
    parser.add_argument("--realizations", type=int, default=2)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    generate_dataset(
        arguments.root,
        arguments.base_seed,
        {"train": arguments.train, "val": arguments.val, "test": arguments.test},
        SquareImageConfig(),
        ComponentTreeKind.MAX_TREE,
        SaltAndPepperNoise(),
        arguments.realizations,
        arguments.workers,
    )


if __name__ == "__main__":
    main()
