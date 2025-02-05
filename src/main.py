# src/main.py

"""
main.py

Main entry point for training and evaluating the MNIST digit recognition model.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import SimpleCNN
from train import train_model, evaluate_model

def main():
    """
    Main function to set up data loaders, initialize the model, train, evaluate, and save the model.
    """
    # Determine if a GPU is available and use it if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10

    # Define data transformations for MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download and prepare the MNIST training and test datasets
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    print("Starting training...")
    model = train_model(model, device, train_loader, criterion, optimizer, num_epochs)

    # Evaluate the model
    print("Evaluating the model...")
    evaluate_model(model, device, test_loader, criterion)

    # Save the trained model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/mnist_cnn.pth")
    print("Model saved to checkpoints/mnist_cnn.pth")

if __name__ == '__main__':
    main()
