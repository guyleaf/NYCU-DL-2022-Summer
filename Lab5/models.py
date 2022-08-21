﻿import torch
import torch.nn as nn

ngf = ndf = 64


class Generator(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(Generator, self).__init__()

        def conv_block(
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            padding: int,
            bias: bool = False,
        ):
            return [
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    bias=bias,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            *conv_block(in_channels, ngf * 8, 4, 1, 0),
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

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        return self.main(torch.concat((x, c), dim=1))


class Discriminator(nn.Module):
    def __init__(self, in_channels: int, n_classes: int):
        super(Discriminator, self).__init__()

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
                )
            ]
            if normalization:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.convs = nn.Sequential(
            # input is (nc) x 64 x 64
            *conv_block(in_channels, ndf, 4, 2, 1, normalization=False),
            # state size. (ndf) x 32 x 32
            *conv_block(ndf, ndf * 2, 4, 2, 1),
            # state size. (ndf*2) x 16 x 16
            *conv_block(ndf * 2, ndf * 4, 4, 2, 1),
            # state size. (ndf*4) x 8 x 8
            *conv_block(ndf * 4, ndf * 8, 4, 2, 1)
        )

        # output networks
        # state size. (ndf*8) x 4 x 4
        self.adversarial_network = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False), nn.Flatten()
        )

        # state size. (ndf*8) x 4 x 4
        self.labels_classifier = nn.Sequential(
            nn.Conv2d(ndf * 8, n_classes, 4, 1, 0, bias=False), nn.Flatten()
        )

    def forward(self, x: torch.Tensor):
        output = self.convs(x)
        real_or_fake = self.adversarial_network(output)
        labels = self.labels_classifier(output)
        return real_or_fake, labels
