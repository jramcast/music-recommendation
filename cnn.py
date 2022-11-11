import torch
from torch import nn
from torchsummary import summary


class CNNNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # Four Convolutional blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                # Treat spectrograms as grayscale images
                in_channels=1,
                # 16 filters in the conv layers
                out_channels=16,
                # Normal value in conv layers
                kernel_size=3,
                stride=1,
                padding=2,
            ),
            # Rectify linear unit
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                # Equal to the number of out channels from prev layer
                in_channels=16,
                # 16 filters in the conv layers
                out_channels=32,
                # Normal value in conv layers
                kernel_size=3,
                stride=1,
                padding=2,
            ),
            # Rectify linear unit
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                # Equal to the number of out channels from prev layer
                in_channels=32,
                # 16 filters in the conv layers
                out_channels=64,
                # Normal value in conv layers
                kernel_size=3,
                stride=1,
                padding=2,
            ),
            # Rectify linear unit
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                # Equal to the number of out channels from prev layer
                in_channels=64,
                # 16 filters in the conv layers
                out_channels=128,
                # Normal value in conv layers
                kernel_size=3,
                stride=1,
                padding=2,
            ),
            # Rectify linear unit
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Flatten the output of the last conv layer
        self.flatten = nn.Flatten()

        # Dense layer
        shape_from_last_conv_block = 128 * 5 * 82
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
