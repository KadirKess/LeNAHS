"""Serve stored observations as model-facing graphs."""

from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from lenahs.dataset.store import NpzDataset
from lenahs.model.adapter import ModelData, adapt_generation_sample

# Position zero is the clean observation of a family. Positions from one
# onward index its noisy observations in stored order.
CLEAN_POSITION = 0


class ObservationDataset(Dataset):
    """One item per observation, loading its family file lazily."""

    def __init__(self, root: str | Path, split: str, include_noisy: bool) -> None:
        self.dataset = NpzDataset.open(root)
        self.split = split
        sample_indices = sorted(
            int(path.stem.removeprefix("sample_"))
            for path in (Path(root) / split).glob("sample_*.npz")
        )
        if not sample_indices:
            raise ValueError(f"Split {split!r} of {root} contains no families.")

        # Every family of a dataset has the same number of noisy observations
        positions = [CLEAN_POSITION]
        if include_noisy:
            noisy_count = len(self.dataset.load(split, sample_indices[0]).noisy)
            positions += range(1, noisy_count + 1)
        self.items = [
            (index, position) for index in sample_indices for position in positions
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, item: int) -> ModelData:
        sample_index, position = self.items[item]
        family = self.dataset.load(self.split, sample_index)
        if position == CLEAN_POSITION:
            sample = family.clean
        else:
            sample = family.noisy[position - 1].sample

        graph = adapt_generation_sample(sample)
        graph.is_clean = torch.tensor(position == CLEAN_POSITION)
        return graph


def make_loader(
    dataset: ObservationDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        generator=generator,
    )
