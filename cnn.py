import math
import torch
from torch import nn
from torchsummary import summary


class CNNNetwork(nn.Module):
    def __init__(
        self,
        input_size=(1, 64, 1292),
        conv_kernel_size=3,
        pooling_kernel_size=2,
        final_conv_channels=128,
    ):
        super().__init__()

        num_channels, num_mels, num_frames = input_size

        # Four Convolutional blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                # Treat spectrograms as grayscale images
                in_channels=num_channels,
                # 16 filters in the conv layers
                out_channels=16,
                # Normal value in conv layers
                kernel_size=conv_kernel_size,
                stride=1,
                padding="same",
            ),
            # Rectify linear unit
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pooling_kernel_size),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                # Equal to the number of out channels from prev layer
                in_channels=16,
                # 16 filters in the conv layers
                out_channels=32,
                # Normal value in conv layers
                kernel_size=conv_kernel_size,
                stride=1,
                padding="same",
            ),
            # Rectify linear unit
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pooling_kernel_size),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                # Equal to the number of out channels from prev layer
                in_channels=32,
                # 16 filters in the conv layers
                out_channels=64,
                # Normal value in conv layers
                kernel_size=conv_kernel_size,
                stride=1,
                padding="same",
            ),
            # Rectify linear unit
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pooling_kernel_size),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                # Equal to the number of out channels from prev layer
                in_channels=64,
                # 16 filters in the conv layers
                out_channels=final_conv_channels,
                # Normal value in conv layers
                kernel_size=conv_kernel_size,
                stride=1,
                padding="same",
            ),
            # Rectify linear unit
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pooling_kernel_size),
        )
        # Output from last conv block (128 out channels, 4 mels, 81

        # Flatten the output of the last conv layer
        self.flatten = nn.Flatten()

        # Calculate the shape of the last conv layer
        # Each pooling kernel divides the size of the original dimension
        num_conv_blocks = 4
        pooling_reduction = pooling_kernel_size**num_conv_blocks
        shape_from_last_conv_block = (
            final_conv_channels
            * math.floor(num_mels / pooling_reduction)
            * math.floor(num_frames / pooling_reduction)
        )

        # Dense layer
        num_classes = 4
        self.linear = nn.Linear(
            in_features=shape_from_last_conv_block, out_features=num_classes
        )

        # Apply softmax for mutliclass classification
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        # Logits: raw values out of the layers
        logits = self.linear(x)
        predictions = self.softmax(logits)
        return predictions


if __name__ == "__main__":
    cnn = CNNNetwork()
    # 1 channel, 64 mel bands, 1292 frames
    data_sample_size = (1, 64, 1292)

    if torch.cuda.is_available():
        cnn = cnn.cuda()

    summary(cnn, data_sample_size)
