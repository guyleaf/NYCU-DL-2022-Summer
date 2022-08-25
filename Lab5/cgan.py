import torch
import torch.nn as nn

ngf = ndf = 64


class CGANGenerator(nn.Module):
    _in_channels: int

    def __init__(self, n_classes: int, z_dim: int, out_channels: int):
        super(CGANGenerator, self).__init__()

        self._z_dim = z_dim
        self._n_classes = n_classes

        def conv_block(
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            padding: int,
            bias: bool = False,
            normalization: bool = True,
        ):
            layers = [
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    bias=bias,
                ),
            ]
            if normalization:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # state size. n_classes x 1
        # like nn.Embedding but for multi-label task
        self.embedding = nn.Sequential(
            nn.ConvTranspose2d(
                n_classes, n_classes, 4, 1, 0, bias=False, groups=n_classes
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(n_classes * 4 * 4, z_dim, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # state size. n_classes x 4 x 4

        self.main = nn.Sequential(
            *conv_block(z_dim * 2, ngf * 8, 4, 1, 0),
            # state size. (ngf*8) x 4 x 4
            *conv_block(ngf * 8, ngf * 4, 4, 2, 1),
            # state size. (ngf*4) x 8 x 8
            *conv_block(ngf * 4, ngf * 2, 4, 2, 1),
            # state size. (ngf*2) x 16 x 16
            *conv_block(ngf * 2, ngf, 4, 2, 1),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, out_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x: torch.Tensor, label: torch.Tensor):
        x = x.view(-1, self._z_dim, 1, 1)
        label = label.view(-1, self._n_classes, 1, 1)

        condition = self.embedding(label).view(-1, self._z_dim, 1, 1)
        return self.main(torch.concat((x, condition), dim=1))


class CGANDiscriminator(nn.Module):
    def __init__(self, in_channels: int, n_classes: int):
        super(CGANDiscriminator, self).__init__()

        self._image_size = 64
        self._in_channels = in_channels
        self._n_classes = n_classes

        def conv_block(
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            padding: int,
            bias: bool = False,
            normalization: bool = True,
        ):
            layers = [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    bias=bias,
                ),
            ]
            if normalization:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.embedding = nn.Sequential(
            nn.ConvTranspose2d(
                n_classes, n_classes, 4, 1, 0, bias=False, groups=n_classes
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(
                n_classes * 4 * 4,
                in_channels * self._image_size**2,
                bias=False,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.convs = nn.Sequential(
            # input is (nc) x 64 x 64
            *conv_block(in_channels * 2, ndf, 4, 2, 1, normalization=False),
            # state size. (ndf) x 32 x 32
            *conv_block(ndf, ndf * 2, 4, 2, 1),
            # state size. (ndf*2) x 16 x 16
            *conv_block(ndf * 2, ndf * 4, 4, 2, 1),
            # state size. (ndf*4) x 8 x 8
            *conv_block(ndf * 4, ndf * 8, 4, 2, 1),
        )

        # output networks
        # state size. (ndf*8) x 4 x 4
        self.adversarial_network = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False), nn.Flatten()
        )

    def forward(self, x: torch.Tensor, label: torch.Tensor):
        label = label.view(-1, self._n_classes, 1, 1)

        condition = self.embedding(label).view(
            -1, self._in_channels, self._image_size, self._image_size
        )
        output = self.convs(torch.concat([x, condition], dim=1))
        real_or_fake = self.adversarial_network(output)
        return real_or_fake
