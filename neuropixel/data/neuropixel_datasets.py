import os
import re
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from glob import glob


class NeuroPixelForSparseCoding(Dataset):
    """
    A PyTorch Dataset for loading NeuroPixel data from .npy files, with functionality to
    split each signal into multiple smaller signals.

    Attributes:
        data_dir (str): Directory containing the .npy files for each channel.
        window_length (int): Length of the window to be used for each signal segment.
        padding (int): Padding length, set equal to window length.
        scale_by (float): Scaling factor for the signal values.
        num_splits (int): Number of segments to split each signal into.
        split_length (int): Length of each split segment.
        overlap (int): Overlap between consecutive segments.
        file_paths (list): List of file paths to the .npy files, filtered by the specified channels.
        temporal_length (int): Length of the signals, assumed to be the same across all channels.

    Methods:
        __len__(): Returns the number of channels (length of file_paths).
        __getitem__(idx): Returns a tensor of split signal segments for the specified channel.
    """

    def __init__(
        self,
        data_dir,
        channels=None,
        window_length=1000,
        scale_by=0.01,
        num_splits=None,
        split_length=None,
        overlap=0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.window_length = window_length
        self.padding = window_length  # Set padding length equal to window length
        self.scale_by = scale_by
        self.overlap = overlap

        # Ensure only one of num_splits or split_length is provided
        assert (num_splits is None) != (
            split_length is None
        ), "Either num_splits or split_length must be provided, but not both."

        # Get all file paths matching the pattern
        all_channels_file_paths = glob(os.path.join(data_dir, "channel_*.npy"))
        all_channels_file_paths = [
            (int(re.search(r"channel_(\d+).npy", file_path).group(1)), file_path)
            for file_path in all_channels_file_paths
        ]
        self.selected_channels = set(
            channels if channels is not None else range(len(all_channels_file_paths))
        )
        self.file_paths = sorted(
            [
                file_path
                for channel_num, file_path in all_channels_file_paths
                if channel_num in self.selected_channels
            ],
            key=lambda x: int(re.search(r"channel_(\d+).npy", x).group(1)),
        )

        # Ensure file names adhere to the naming convention
        for file_path in self.file_paths:
            assert re.match(
                r".*channel_\d+\.npy$", file_path
            ), f"File {file_path} does not match the naming convention 'channel_i.npy'"

        if not self.file_paths:
            raise ValueError(f"No .npy files found in directory: {data_dir}")

        # Load the full length of data for the temporal dimension
        self.temporal_length = np.load(self.file_paths[0], mmap_mode="r").shape[0]

        # Validate that all files have the same length
        for file_path in self.file_paths:
            assert (
                np.load(file_path, mmap_mode="r").shape[0] == self.temporal_length
            ), "All files must have the same length."

        # Calculate num_splits or split_length based on the provided parameter
        if num_splits is not None:
            self.num_splits = num_splits
            self.split_length = (
                self.temporal_length - self.overlap * (self.num_splits - 1)
            ) // self.num_splits
        else:
            self.split_length = split_length
            self.num_splits = (self.temporal_length - self.overlap) // (
                self.split_length - self.overlap
            )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        signal = np.load(self.file_paths[idx], mmap_mode="r")
        signal = self.scale_by * signal
        splits = self._split_signal(signal)
        return torch.from_numpy(splits).view(len(splits), -1)

    def _split_signal(self, signal):
        """
        Splits the signal into multiple segments with the specified overlap.

        Args:
            signal (numpy.ndarray): The input signal to be split.

        Returns:
            numpy.ndarray: An array of split signal segments.
        """
        splits = []
        current_idx = 0

        while current_idx + self.split_length <= self.temporal_length:
            splits.append(signal[current_idx : current_idx + self.split_length])
            current_idx += self.split_length - self.overlap

        if current_idx < self.temporal_length:
            last_split = signal[current_idx:]
            if len(last_split) < self.split_length:
                last_split = np.pad(
                    last_split, (0, self.split_length - len(last_split))
                )
            splits.append(last_split)

        return np.stack(splits, axis=0)


def collate_fn_for_sparse_coding_ds(batch):
    """
    Custom collate function for DataLoader.

    Args:
        batch: Batch of split signals.

    Returns:
        Concatenated tensor of the batch along the first dimension.
    """
    return torch.cat(batch, dim=0)


class NeuroPixelForDictionaryUpdate(IterableDataset):
    """
    A PyTorch IterableDataset for loading NeuroPixel data from .npy files, with functionality to
    randomly sample channels and temporal windows for dictionary updates.

    Attributes:
        data_dir (str): Directory containing the .npy files for each channel.
        window_length (int): Length of the window to be used for each signal segment.
        padding (int): Padding length, set equal to window length.
        scale_by (float): Scaling factor for the signal values.
        file_paths (list): List of file paths to the .npy files, filtered by the specified channels.
        temporal_length (int): Length of the signals, assumed to be the same across all channels.

    Methods:
        __iter__(): Yields a randomly selected and scaled signal segment for dictionary updates.
    """

    def __init__(
        self,
        data_dir,
        channels=None,
        window_length=1000,
        scale_by=0.01,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.window_length = window_length
        self.padding = window_length  # Set padding length equal to window length
        self.scale_by = scale_by

        # Get all file paths matching the pattern
        all_channels_file_paths = glob(os.path.join(data_dir, "channel_*.npy"))
        all_channels_file_paths = [
            (int(re.search(r"channel_(\d+).npy", file_path).group(1)), file_path)
            for file_path in all_channels_file_paths
        ]
        self.selected_channels = set(
            channels if channels is not None else range(len(all_channels_file_paths))
        )
        self.file_paths = sorted(
            [
                file_path
                for channel_num, file_path in all_channels_file_paths
                if channel_num in self.selected_channels
            ],
            key=lambda x: int(re.search(r"channel_(\d+).npy", x).group(1)),
        )

        # Ensure file names adhere to the naming convention
        for file_path in self.file_paths:
            assert re.match(
                r".*channel_\d+\.npy$", file_path
            ), f"File {file_path} does not match the naming convention 'channel_i.npy'"

        if not self.file_paths:
            raise ValueError(f"No .npy files found in directory: {data_dir}")

        # Load the full length of data for the temporal dimension
        self.temporal_length = np.load(self.file_paths[0], mmap_mode="r").shape[0]

        # Validate that all files have the same length
        for file_path in self.file_paths:
            assert (
                np.load(file_path, mmap_mode="r").shape[0] == self.temporal_length
            ), "All files must have the same length."

    def __iter__(self):
        while True:
            channel_idx = np.random.randint(0, len(self.file_paths))

            # Load the selected channel's signal
            signal = np.load(self.file_paths[channel_idx], mmap_mode="r")

            # Randomly select a starting position for the temporal window within the padded range
            start_idx = np.random.randint(
                0, max(1, self.temporal_length - self.window_length + 1)
            )
            end_idx = start_idx + self.window_length

            # Trim the signal to the selected temporal window
            trimmed_signal = signal[start_idx:end_idx]
            assert (
                len(trimmed_signal) == self.window_length
            ), f"trimmed_signal length: {len(trimmed_signal)} != window_length: {self.window_length}"
            scaled_trimmed_signal = self.scale_by * trimmed_signal
            yield torch.from_numpy(scaled_trimmed_signal).view(1, self.window_length)


if __name__ == "__main__":
    data_dir = "C:/Users/yoav.h/research/crsae_on_neuropixel/data/data_files/200k_samp/by_channel"

    # Test NeuroPixelForSparseCoding Dataset
    print("Testing NeuroPixelForSparseCoding Dataset")
    sparse_coding_dataset = NeuroPixelForSparseCoding(
        data_dir, window_length=1000, scale_by=0.01, split_length=40000, overlap=100
    )
    sparse_coding_dataloader = DataLoader(
        sparse_coding_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn_for_sparse_coding_ds,
    )
    for batch in sparse_coding_dataloader:
        print(batch.shape)
        break

    # Test NeuroPixelForDictionaryUpdate IterableDataset
    print("Testing NeuroPixelForDictionaryUpdate IterableDataset")
    dictionary_update_dataset = NeuroPixelForDictionaryUpdate(
        data_dir, window_length=1000, scale_by=0.01
    )
    dictionary_update_dataloader = DataLoader(
        dictionary_update_dataset, batch_size=32, shuffle=False
    )
    for batch in dictionary_update_dataloader:
        print(batch.shape)
        break
