import numpy as np
import scipy.io
from pathlib import Path

root_dir = Path("C:/Users/yoav.h/research/crsae_on_neuropixel")
data_root_dir = root_dir / "data"
experiment_root_dir = root_dir / "experiments"


def preprocess_and_save_by_channel(
    mat_file_path, output_dir, channels=None, subtract_median=True
):
    if not Path(mat_file_path).exists():
        raise ValueError(f"File not found: {mat_file_path}")

    # Load the data
    data = scipy.io.loadmat(mat_file_path)["dataArray"]

    # Subtract the median if specified
    if subtract_median:
        data = data - np.median(data, axis=0, keepdims=True)

    # Select specified channels if provided, otherwise use all channels
    if channels is None:
        channels = range(data.shape[0])

    # Create the output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save each channel to a separate .npy file
    for channel in channels:
        channel_data = data[channel, :].astype(np.float32)
        output_file = Path(output_dir) / f"channel_{channel}.npy"
        np.save(output_file, channel_data)


# Paths and channels
input_mat_file = data_root_dir / "data_files/200k_samp/data_for_200k_samp.mat"
output_directory = data_root_dir / "data_files/200k_samp/by_channel"
selected_channels = (
    None  # Example channels, set to None if you want to process all channels
)
# Preprocess and save
preprocess_and_save_by_channel(input_mat_file, output_directory, selected_channels)
