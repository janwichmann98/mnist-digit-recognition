# src/model.py

"""
model.py

Defines the neural network model for MNIST digit recognition.
"""

import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    Simple Convolutional Neural Network for MNIST Digit Recognition.
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First convolutional layer: 1 input channel, 32 output channels, kernel size 3, padding to preserve spatial dimensions.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # Second convolutional layer: 32 input channels, 64 output channels, kernel size 3, padding to preserve spatial dimensions.
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Fully connected layer 1: from flattened feature map to 128 neurons.
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # Fully connected layer 2: from 128 neurons to 10 output classes.
        self.fc2 = nn.Linear(128, 10)
        # Dropout layer to reduce overfitting.
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        """
        Forward pass of the neural network.
        """
        # Apply first convolution, followed by ReLU activation and 2x2 max pooling.
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        # Apply second convolution, followed by ReLU activation and 2x2 max pooling.
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        # Flatten the output for the fully connected layers.
        x = x.view(x.size(0), -1)
        # Apply first fully connected layer with ReLU activation.
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # Apply output layer.
        x = self.fc2(x)
        return x
