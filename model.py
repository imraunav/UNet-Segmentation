import torch
from torch import nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, src):
        return self.conv(src)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        features=[64, 128, 256, 512],
    ) -> None:
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # downsampling in UNet
        for n_feature in features:
            self.downs.append(DoubleConv(in_channels, n_feature))
            in_channels = n_feature

        # upsampling in UNet
        for n_feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(n_feature * 2, n_feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(n_feature * 2, n_feature))

        # bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.out = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        skip_connections = skip_connections[::-1]

        x = self.bottleneck(x)

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i // 2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, skip_connection.shape[-2:], antialias=False)
            x = torch.cat([skip_connection, x], dim=1)
            x = self.ups[i + 1](x)
        return self.out(x)


def test():
    x = torch.randn(3, 3, 163, 163)
    model = UNet(3, 3)
    preds = model(x)
    print(x.shape, preds.shape)
    assert x.shape == preds.shape


if __name__ == "__main__":
    test()
