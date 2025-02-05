# src/train.py

"""
train.py

Contains functions for training and evaluating the MNIST digit recognition model.
"""

import torch
#import torch.nn as nn
#import torch.optim as optim

def train_model(model, device, train_loader, criterion, optimizer, num_epochs):
    """
    Train the model on the training dataset.

    Args:
        model (nn.Module): The neural network model.
        device (torch.device): Device to run the training on (CPU or GPU).
        train_loader (DataLoader): DataLoader for the training data.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer for model parameters.
        num_epochs (int): Number of epochs for training.

    Returns:
        model (nn.Module): The trained model.
    """
    model.train()
    for epoch in range(1, num_epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, target)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100.0

        print(f"Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    return model

def evaluate_model(model, device, test_loader, criterion):
    """
    Evaluate the model on the test dataset.

    Args:
        model (nn.Module): The trained neural network model.
        device (torch.device): Device to run evaluation on.
        test_loader (DataLoader): DataLoader for the test data.
        criterion (nn.Module): Loss function.

    Returns:
        test_loss (float): Average loss on the test dataset.
        test_accuracy (float): Accuracy percentage on the test dataset.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # Disable gradient computation for evaluation
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            outputs = model(data)
            loss = criterion(outputs, target)

            running_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    test_loss = running_loss / total
    test_accuracy = (correct / total) * 100.0
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    return test_loss, test_accuracy
