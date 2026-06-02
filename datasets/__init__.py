from .base import DATASET_REGISTRY, BaseDataset, build_dataset_splits, write_dataset_splits
from .books import AmazonBooksDataset
from .ml_32m import MovieLens32MDataset

DATASET_REGISTRY.update(
    {
        "movielens": MovieLens32MDataset,
        "movielens32m": MovieLens32MDataset,
        "amazon": AmazonBooksDataset,
        "amazon_books": AmazonBooksDataset,
    }
)

__all__ = [
    "AmazonBooksDataset",
    "BaseDataset",
    "DATASET_REGISTRY",
    "MovieLens32MDataset",
    "build_dataset_splits",
    "write_dataset_splits",
]
