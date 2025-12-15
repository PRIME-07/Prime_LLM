import torch
from torch.utils.data import DataLoader

def create_dataloader(dataset, batch_size=8, shuffle=True, num_workers=2):
    """
    Wraps a LyricsDataset into a PyTorch DataLoader.

    Args:
        dataset: LyricsDataset instance
        batch_size: number of sequences per batch
        shuffle: shuffle training order
        num_workers: workers to speed up batching

    Returns:
        DataLoader object
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # ensures uniform batch sizes
    )
