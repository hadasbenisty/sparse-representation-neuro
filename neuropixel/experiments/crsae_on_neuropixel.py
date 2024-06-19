import os
from datetime import datetime
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader, IterableDataset

from neuropixel.data.neuropixel_datasets import NeuroPixelForDictionaryUpdate
from crsae.crsae_model import CRsAE1D
from crsae.model_trainer import ModelTrainer


def load_config(config_path):
    """
    Load configuration from a YAML file.

    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def validate_config(config):
    """
    Validate that the configuration contains all required keys.
    """
    required_keys = [
        "model_params",
        "train_params",
        "dataset_params",
        "experiment_name",
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")


def main(config_path):
    """
    Main function to run the training process.

    Args:
        config_path (str): Path to the configuration file.
    """
    config = load_config(config_path)
    validate_config(config)

    model_params = config["model_params"]
    train_params = config["train_params"]
    dataset_params = config["dataset_params"]
    experiment_name = config["experiment_name"]

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model_params["device"] = device  # Ensure device is correctly set at runtime

    model_params["data_signal_length"] = dataset_params["split_length"]
    model = CRsAE1D(**model_params)

    # Initialize NeuroPixelForDictionaryUpdate Dataset and DataLoader
    stochastic_data_set = NeuroPixelForDictionaryUpdate(
        data_dir=dataset_params["data_dir"],
        channels=dataset_params["channels"],
        window_length=dataset_params["window_length"],
        scale_by=dataset_params["scale_by"],
    )
    stochastic_data_loader = DataLoader(
        stochastic_data_set,
        batch_size=train_params["batch_size"],
        shuffle=False,
        num_workers=train_params.get("num_workers", 0),
        prefetch_factor=train_params.get("prefetch_factor", 2),
        persistent_workers=train_params.get("persistent_workers", False),
    )

    output_dir = (
        Path(config["experiments_dir"])
        / f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if isinstance(stochastic_data_set, IterableDataset):
        steps_per_epoch = round(
            len(dataset_params["channels"])
            * (stochastic_data_set.temporal_length // dataset_params["window_length"])
            / train_params["batch_size"]
        )
        train_params["steps_per_epoch"] = steps_per_epoch

    print(f"using {train_params['steps_per_epoch']} steps in epochs")

    # Save a copy of the experiment configuration as a YAML file in the output directory
    config_save_path = output_dir / "experiment_config.yaml"
    with open(config_save_path, "w") as config_file:
        yaml.dump(config, config_file)
    print(f"Configuration saved to {config_save_path}")

    # Initialize and run the trainer
    trainer = ModelTrainer(model, model_params, train_params, device, experiment_name)
    trainer.fit(stochastic_data_loader, output_dir)


if __name__ == "__main__":
    main("experiment_config.yaml")
