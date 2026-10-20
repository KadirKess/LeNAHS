# LeNAHS

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KadirKess/LeNAHS/blob/main/notebooks/train.ipynb)

LeNAHS (Learning Noise-Aware Hierarchical Structures) is a research project. It
asks whether a graph neural network can recover the component tree of a clean
image when it only sees a noisy version of that image.

The model starts from the region adjacency graph (RAG) of an image. Each node is
a connected region of constant gray level, and edges join regions that touch.
Noise breaks an image into many small regions that do not belong to its real
structure. LeNAHS trains a model to say, for each region of the noisy image,
where it belongs in the clean image's hierarchy.

## Layout

```
src/lenahs/
  component_tree/   RAG and max/min-tree records
    reference/      readable construction of the RAG and component tree
    higra/          Higra-based construction and conversion to one parent per node
  generation/       synthetic images, noise models, seeds and correction targets
  dataset/          on-disk sample storage, manifests and validation
  model/            graph adapter, recurrent network, parent and action heads
  training/         losses, metrics, epoch loop and checkpointing
scripts/            dataset generation and training entry points
notebooks/          Colab notebook that generates a dataset and trains
tests/              tests mirroring the package layout
```

## Getting started

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync
uv run pytest
```

To generate a salt-and-pepper dataset and train on it:

```bash
uv run python scripts/generate_preliminary_dataset.py dataset
uv run python scripts/train_preliminary.py dataset run/main --max-epochs 100
```

Training uses a CUDA GPU when one is available, then Apple's MPS backend, then
the CPU. The Colab notebook runs the same two steps on a hosted GPU.

## License

MIT. See [LICENSE](LICENSE).
