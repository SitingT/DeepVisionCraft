import torch
from torch import nn
from torchsummary import summary


# Define the LeNet neural network architecture by inheriting from nn.Module
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        # First convolutional layer: Takes a 1-channel (grayscale) input, applies 6 filters with a 5x5 kernel, and adds padding of 2
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6,
                            kernel_size=5, padding=2)

        # Sigmoid activation function
        self.sig = nn.Sigmoid()

        # First pooling layer: Applies average pooling with a 2x2 kernel and a stride of 2
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Second convolutional layer: Takes 6 input channels, applies 16 filters with a 5x5 kernel
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        # Second pooling layer: Applies average pooling with a 2x2 kernel and a stride of 2
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Flatten layer to convert 2D feature maps to 1D vector
        self.flatten = nn.Flatten()

        # First fully connected layer: Takes 400 input features and outputs 120 features
        self.f5 = nn.Linear(400, 120)

        # Second fully connected layer: Takes 120 input features and outputs 84 features
        self.f6 = nn.Linear(120, 84)

        # Output layer: Takes 84 input features and outputs 10 features (for classification)
        self.f7 = nn.Linear(84, 10)

    def forward(self, x):
        # Forward pass through the network
        x = self.sig(self.c1(x))   # Apply first convolution and activation
        x = self.s2(x)             # Apply first pooling
        x = self.sig(self.c3(x))   # Apply second convolution and activation
        x = self.s4(x)             # Apply second pooling
        x = self.flatten(x)        # Flatten for fully connected layers
        x = self.f5(x)             # First fully connected layer
        x = self.f6(x)             # Second fully connected layer
        x = self.f7(x)             # Output layer
        return x


if __name__ == "__main__":
    # Check if GPU is available; otherwise, use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the LeNet model and move it to the selected device
    model = LeNet().to(device)

    # Print the model summary, displaying input/output shapes and parameter counts for each layer
    print(summary(model, (1, 28, 28)))
