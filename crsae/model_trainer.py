import os
import yaml
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from datetime import datetime


class ModelTrainer:
    def __init__(
        self,
        model,
        model_params,
        train_params,
        device,
        experiment_name="experiment",
    ):
        """
        Initialize the ModelTrainer.

        Parameters:
        model (torch.nn.Module): The model to be trained.
        model_params (dict): Dictionary of model parameters.
        train_params (dict): Dictionary of training parameters.
        device (str): Device to use for training (e.g., 'cpu', 'cuda:0').
        experiment_name (str, optional): Name of the experiment. Default is 'experiment'.
        """
        self.model = model
        self.model_params = model_params
        self.train_params = train_params
        self.device = device
        self.experiment_name = experiment_name
        self.criterion = torch.nn.MSELoss().to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.train_params["lr"], eps=1e-4
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.2,
            patience=8,
            threshold=1e-4,
            min_lr=1e-7,
        )
        self.patience = self.train_params.get("patience", 5)
        self.dict_opt_steps = self.train_params.get("dict_opt_steps", 10)

        if self.model_params.get("update_L", False):
            self.update_L_frequency = self.train_params.get("update_L_frequency", 10)

    def fit(self, data_loader, output_dir):
        """
        Fit the model to the data.

        Parameters:
        data_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        output_dir (str): Directory to save the outputs.
        """
        create_output_directories(output_dir)
        best_loss = float("inf")
        epochs_no_improve = 0
        filter_plot_files = []

        # Plot initial filters
        plot_file = plot_and_save_filters(
            self.model, 0, self.experiment_name, output_dir
        )
        filter_plot_files.append(plot_file)

        for epoch in range(self.train_params["num_epochs"]):
            print(f"Epoch {epoch + 1}/{self.train_params['num_epochs']}")
            if epoch % self.update_L_frequency == 0:
                self.model.L = self.model.calculate_L_from_filters()

            initial_H = self.model.get_param("filters")
            epoch_loss = self.dictionary_optimization(data_loader, initial_H)
            print(
                f"Epoch [{epoch + 1}/{self.train_params['num_epochs']}] loss: {epoch_loss:.4f}"
            )
            plot_file = plot_and_save_filters(
                self.model, epoch, self.experiment_name, output_dir
            )
            filter_plot_files.append(plot_file)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(
                    self.model.state_dict(),
                    output_dir
                    / f"{self.experiment_name}_loss_{round(best_loss, 2)}_best.pth",
                )
                print(f"New best model saved with loss {best_loss:.4f}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

        torch.save(self.model.state_dict(), output_dir / f"{self.experiment_name}.pth")
        print(f"Final model saved.")
        print(f"Best loss: {best_loss:.4f}")
        create_video_from_plots(filter_plot_files, output_dir, self.experiment_name)
        self.save_config(output_dir)

    def dictionary_optimization(self, data_loader, initial_H):
        """
        Perform dictionary optimization.

        Parameters:
        data_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        initial_H (torch.Tensor): Initial filters, used to make sure x remains set and optimization is not joint.

        Returns:
        float: Average step loss.
        """
        self.model.train()
        step_losses = []
        for step, y_batch in enumerate(
            tqdm(data_loader, desc="Dictionary Optimization")
        ):
            y_batch = y_batch.to(self.device)
            self.optimizer.zero_grad()

            # Step 1: Compute the fixed x by using the initial H_0.
            sparse_x = self.model.encoder(y_batch, H=initial_H)

            # Step 2: Compute y_hat with current H_t
            y_hat_batch = self.model.decoder(sparse_x)

            # Step 3: Compute loss between y and y_hat
            step_loss = self.criterion(y_batch, y_hat_batch)

            # Step 4: Compute gradient of the loss w.r.t H_t
            step_loss.backward()

            # Step 5: Update H_t using the optimizer
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.model.normalize_filters()
            step_losses.append(step_loss.item())
            self.scheduler.step(step_loss)

            if step >= self.dict_opt_steps:
                break

        avg_step_loss = torch.mean(torch.tensor(step_losses, device=self.device))
        return avg_step_loss

    def save_config(self, output_dir):
        """
        Save the configuration of the experiment.

        Parameters:
        output_dir (str): Directory to save the configuration.
        """
        config = {
            "experiment_name": self.experiment_name,
            "model_params": self.model_params,
            "train_params": self.train_params,
            "device": str(self.device),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        config_path = output_dir / f"{self.experiment_name}_config.yaml"
        with open(config_path, "w") as config_file:
            yaml.dump(config, config_file)
        print(f"Configuration saved to {config_path}")


# Helper functions
def create_output_directories(output_dir):
    """
    Create output directories if they do not exist.

    Parameters:
    output_dir (str): Directory to create.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_dir / "filter_plots"):
        os.makedirs(output_dir / "filter_plots")


def create_video_from_plots(plot_files, output_dir, model_name):
    """
    Create a video from saved filter plots.

    Parameters:
    plot_files (list): List of plot file paths.
    output_dir (str): Directory to save the video.
    model_name (str): Name of the model.
    """
    with imageio.get_writer(output_dir / f"{model_name}_filters.mp4", fps=1) as writer:
        for plot_file in plot_files:
            image = imageio.imread(plot_file)
            writer.append_data(image)


def plot_and_save_filters(model, epoch, model_name, output_dir):
    """
    Plot and save the filters of the model.

    Parameters:
    model (torch.nn.Module): The model containing the filters.
    epoch (int): Current epoch number.
    model_name (str): Name of the model.
    output_dir (str): Directory to save the plots.

    Returns:
    str: Path to the saved plot file.
    """
    print(f"Plotting and saving filters for epoch {epoch + 1}")
    filters = (
        model.get_param("filters").cpu().detach().numpy()
    )  # Ensure filters are moved to CPU for plotting
    num_filters = filters.shape[0]
    fig, axs = plt.subplots(num_filters, 1, figsize=(10, num_filters * 2))
    for i in range(num_filters):
        axs[i].plot(filters[i, 0, :])
        axs[i].set_title(f"Filter {i + 1}")
    plt.tight_layout()
    plot_file = (
        output_dir / "filter_plots" / f"{model_name}_epoch_{epoch + 1}_filters.png"
    )
    plt.savefig(plot_file)
    plt.close(fig)
    print(f"Filters saved to {plot_file}")
    return plot_file


if __name__ == "__main__":
    pass
