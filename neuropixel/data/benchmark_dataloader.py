""""
This is meant to find the optimal parameters for data loading. It compares different parameters over different batch sizes. 
"""

import time
import torch
from torch.utils.data import DataLoader
from neuropixel_datasets import (
    NeuroPixelForSparseCoding,
    NeuroPixelForDictionaryUpdate,
    collate_fn_for_sparse_coding_ds,
)


def benchmark_dataloader(dataloader, device, name, batch_num):
    start_time = time.time()
    for i, batch in enumerate(dataloader):
        if i >= batch_num:  # Process first 10 batches to get a good estimate
            break
        batch = batch.to(device)
        # Simulate training workload
        with torch.no_grad():
            output = batch * 2  # Simple operation to simulate GPU workload
    end_time = time.time()
    print(f"{name} - time_taken: {end_time - start_time:.2f}s")


if __name__ == "__main__":
    data_dir = "C:/Users/yoav.h/research/crsae_on_neuropixel/data/data_files/200k_samp/by_channel"
    window_length = 200000
    channels = [160, 161]
    scale_by = 0.1
    split_length = 40000
    overlap = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the datasets
    sparse_coding_dataset = NeuroPixelForSparseCoding(
        data_dir=data_dir,
        channels=channels,
        window_length=window_length,
        scale_by=scale_by,
        split_length=split_length,
        overlap=overlap,
    )

    dictionary_update_dataset = NeuroPixelForDictionaryUpdate(
        data_dir=data_dir,
        channels=channels,
        window_length=window_length,
        scale_by=scale_by,
    )

    num_workers_list = [0, 2, 4, 8]  # Adjust based on your CPU cores
    prefetch_factor_list = [2, 4, 8]
    batch_size_list = [2, 4, 8, 16]  # Try different batch sizes

    for num_workers in num_workers_list:
        prefetch_factor_list_ = [None] if num_workers == 0 else prefetch_factor_list
        for prefetch_factor in prefetch_factor_list_:
            for batch_size in batch_size_list:
                # Benchmark NeuroPixelForSparseCoding DataLoader
                sparse_coding_dataloader = DataLoader(
                    sparse_coding_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=collate_fn_for_sparse_coding_ds,
                    num_workers=num_workers,
                    prefetch_factor=prefetch_factor,
                    persistent_workers=False,
                )
                benchmark_dataloader(
                    sparse_coding_dataloader,
                    device,
                    f"SparseCoding (num_workers={num_workers}, prefetch_factor={prefetch_factor}, batch_size={batch_size})",
                    50,
                )

                # Benchmark NeuroPixelForDictionaryUpdate DataLoader
                dictionary_update_dataloader = DataLoader(
                    dictionary_update_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    prefetch_factor=prefetch_factor,
                    persistent_workers=False,
                )
                benchmark_dataloader(
                    dictionary_update_dataloader,
                    device,
                    f"DictionaryUpdate (num_workers={num_workers}, prefetch_factor={prefetch_factor}, batch_size={batch_size})",
                    50,
                )
