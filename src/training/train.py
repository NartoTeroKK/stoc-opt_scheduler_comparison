import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml
import sys

# Add the src directory to the path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.logistic_regression import LogisticRegression
from models.mlp import MLP


def load_processed_split(dataset_name: str, split: str, root='data'):
    """
    Load a processed split from the saved .pt files.
    Args:
        dataset_name: either 'breast_cancer' or 'synthetic'
        split: one of 'train', 'val', 'test'
        root: root directory where the processed data is saved
    Returns:
        TensorDataset containing the features and labels
    """
    fname = f'{dataset_name}_{split}.pt'
    fpath = os.path.join(root, 'processed', fname)
    X, y = torch.load(fpath)
    return TensorDataset(X, y)


def get_model(model_name: str, input_dim: int, output_dim: int = 2):
    """
    Return a model instance based on the model name.
    """
    if model_name == 'logistic':
        return LogisticRegression(input_dim, output_dim)
    elif model_name == 'mlp':
        # For simplicity, we'll use a fixed hidden layer structure for MLP
        # In a real experiment, this would come from config
        hidden_dims = [64, 32]  # two hidden layers
        return MLP(input_dim, hidden_dims, output_dim)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_optimizer(optimizer_name: str, model_params, lr: float):
    """
    Return an optimizer instance based on the optimizer name.
    """
    if optimizer_name == 'sgd':
        return torch.optim.SGD(model_params, lr=lr)
    elif optimizer_name == 'adam':
        return torch.optim.Adam(model_params, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_scheduler(scheduler_name: str, optimizer, config: dict):
    """
    Return a scheduler instance based on the scheduler name.
    For now, we only support 'none' (returns None).
    In future versions, we will implement the schedulers.
    """
    if scheduler_name == 'none':
        return None
    else:
        # Placeholder for future implementation
        raise NotImplementedError(f"Scheduler {scheduler_name} not implemented yet")


def main():
    parser = argparse.ArgumentParser(description='Training script for scheduler comparison')
    parser.add_argument('--optimizer', type=str, default='sgd', help='Optimizer to use (sgd or adam)')
    parser.add_argument('--scheduler', type=str, default='none', help='Scheduler to use (none, cosine, exponential, clr, onecycle)')
    parser.add_argument('--model', type=str, default='logistic', help='Model to use (logistic or mlp)')
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml', help='Path to config YAML file')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs to train (overrides config)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (overrides config)')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override config with command line arguments if provided
    if args.seed is not None:
        config['seed'] = args.seed
    if args.epochs is not None:
        config['epochs'] = args.epochs

    # Set random seeds for reproducibility
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the breast cancer training data (for verification)
    # We use the training split to verify that the data loader works
    train_dataset = load_processed_split('breast_cancer', 'train', root='data')
    print(f"Loaded breast cancer training dataset with {len(train_dataset)} samples")

    # Create DataLoader
    batch_size = config.get('batch_size', 128)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Get input dimension from the dataset
    # We assume the dataset is not empty
    sample_features, _ = train_dataset[0]
    input_dim = sample_features.shape[0]
    print(f"Input dimension: {input_dim}")

    # Initialize model
    model = get_model(args.model, input_dim)
    model = model.to(device)

    # Initialize optimizer
    lr = config.get('lr_initial', 0.01)
    optimizer = get_optimizer(args.optimizer, model.parameters(), lr)

    # Initialize scheduler
    scheduler = get_scheduler(args.scheduler, optimizer, config)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop
    epochs = config.get('epochs', 30)
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')

        # Scheduler step (if scheduler is not None)
        if scheduler is not None:
            scheduler.step()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        print(f'Epoch [{epoch+1}/{epochs}] Completed: Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

    print("Training finished!")


if __name__ == '__main__':
    main()