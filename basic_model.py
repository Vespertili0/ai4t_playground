import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm


## Basic Data Preparation Class
class DataPrep(data.Dataset):
    def __init__(self, features: np.array, targets: np.array) -> tuple:
        """
        Args:
            features (numpy.ndarray or torch.Tensor): Feature data (inputs).
            targets (numpy.ndarray or torch.Tensor): Target data (labels).
            transform (callable, optional): Optional transform to apply to the features.
        """
        self.features = torch.tensor(features, dtype=torch.float32) if not isinstance(features, torch.Tensor) else features
        self.targets = torch.tensor(targets, dtype=torch.float32) if not isinstance(targets, torch.Tensor) else targets

    def __len__(self):
        # Returns the total number of samples in the dataset
        return len(self.features)

    def __getitem__(self, index):
        # Retrieves a sample from the dataset at the specified index
        feature = self.features[index]
        target = self.targets[index]

        return feature, target


def create_dataloaders_from_prep(
        dataset: torch.utils.data.Dataset,
        batch_size=32,
        validation_split=0.2, 
        shuffle=True
        )-> tuple:
    """
    Splits a dataset into training and validation sets, and returns their torch.utils.data.DataLoader objects.
    
    Args:
        dataset (torch.utils.data.Dataset): PyTorch-dataset object of full dataset.
        batch_size (int): Batch size for the DataLoaders.
        validation_split (float): Proportion of data to use for validation.
        shuffle (bool): Whether to shuffle the data during training.

    Returns:
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
    """
    # Calculate sizes for training and validation sets
    val_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - val_size

    # Split the dataset
    train_dataset, val_dataset = data.random_split(dataset, [train_size, val_size])

    # Create DataLoaders for training and validation
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def calculate_batch_losses(
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader, 
        loss_fn: torch.nn.Module, 
        optimiser: torch.optim.Optimizer,
        mode='train'
) -> float:
    """
    Calculates the average loss for batches of data. In 'train' mode, model parameters are updated.
    Args:
        model: PyTorch model to be trained or validated.
        data_loader: DataLoader for the dataset (training or validation).
        loss_fn: Loss function.
        optimiser: Optimizer for training.
        mode: 'train' or 'val' to indicate the mode of operation.
    Returns: float
        average_loss
    
    """
    assert mode in ['train', 'val'], f"Invalid mode: {mode}. Choose 'train' or 'val'."
   
    total_loss, total_samples = 0, 0

    grad_mode = torch.enable_grad() if mode == 'train' else torch.no_grad()

    with grad_mode:
        for batch_X, batch_y in data_loader:
            loss = loss_fn(model(batch_X), batch_y)

            if mode == 'train':
                loss.backward()
                optimiser.step()
                optimiser.zero_grad()

            total_loss += loss.item() * len(batch_X)
            total_samples += len(batch_X)

    avg_loss = total_loss / total_samples
    return avg_loss


def run_epoch(
        model: torch.nn.Module, 
        train_loader: torch.utils.data.DataLoader, 
        val_loader: torch.utils.data.DataLoader, 
        loss_fn: torch.nn.Module, 
        optimiser: torch.optim.Optimizer 
    ) -> tuple:
    """
    Run a single epoch of training and validation.
    Args:
        model: PyTorch model to be trained and validated.
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        loss_fn: Loss function.
        optimiser: Optimizer for training.
    Returns: tuple
        (avg_train_loss, avg_val_loss): Average training and validation loss for the epoch.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    # run training step
    avg_train_loss = calculate_batch_losses(
        model, 
        train_loader, 
        loss_fn, 
        optimiser, 
        mode='train'
    )
    # run validation step
    avg_val_loss = calculate_batch_losses(
        model, 
        val_loader, 
        loss_fn, 
        optimiser=None, 
        mode='val'
    )
    
    return avg_train_loss, avg_val_loss


def fit_model(
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        num_epochs: int = 10,
        n_step_report: int = 5,
    ) -> None:
    """
    Train and validate a model over multiple epochs.
    
    Args:
        model: PyTorch model to be trained and validated.
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        loss_fn: Loss function.
        optimiser: Optimizer for training.
        num_epochs: Number of epochs to train the model.
        n_step_report: Frequency of reporting training and validation loss.
    Returns:
        None
    """
    for epoch in tqdm(range(1, num_epochs + 1), desc='Training Progress'):
        avg_train_loss, avg_val_loss = run_epoch(
            model, 
            train_loader, 
            val_loader, 
            loss_fn, 
            optimiser
        )
        if epoch % n_step_report == 0:
            print(f"Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")