"""Implementation of the convolutional part of the neural network for text recognition"""
from torch import nn


class ConvolutionalEncoder(nn.Module):
    def __init__(self, n_channels: int):
        """Encodes an image into a representation using a fully convolutional
        architecture.

        Refer to the paper for more information.

        Parameters:
        - n_channels: the number of channel in the input image.
        """
        super().__init__()

        self.cnn = nn.Sequential(
            # First layer
            ConvReLU(
                in_channels=n_channels,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            # Second layer
            ConvReLU(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            # Third layer
            ConvReLU(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                batch_norm=True,
            ),
            # Fourth layer
            ConvReLU(
                in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            # Fifth layer
            ConvReLU(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
                batch_norm=True,
            ),
            # Sixth layer
            ConvReLU(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
                batch_norm=True,
            ),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            # Seventh layer
            ConvReLU(
                in_channels=512, out_channels=512, kernel_size=2, stride=1, padding=0
            ),
        )

    def forward(self, x):
        """x is an image tensor of shape:

        batch_size, channels, height, width
        """

        HEIGHT_IX = 2
        assert x.shape[HEIGHT_IX] == 32  # Or it won't work
        return self.cnn(x)


class ConvReLU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        batch_norm=False,
        leaky=None,
    ):
        """Slighlty different implementation of a regular convolutional
        layer, with the addition of (optional) batch normalization and leaky relu.
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        if leaky is not None:
            # TODO: what is inplace?
            self.relu = nn.LeakyReLU(leaky, inplace=True)
        else:
            self.relu = nn.ReLU(True)

        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None

    def forward(self, x):
        maps = self.conv(x)
        if self.bn is not None:
            maps = self.bn(maps)
        return self.relu(maps)
