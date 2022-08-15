from typing import List, Tuple
import torch
import torch.nn as nn


class VGGLayer(nn.Module):
    layer: nn.Sequential

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class Encoder(nn.Module):
    dconvs: nn.ModuleList
    dlast_conv: nn.Module
    max_pool: nn.Module

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        layers = []
        layers.append(
            nn.Sequential(
                VGGLayer(in_channels, 64),
                VGGLayer(64, 64),
            )
        )
        layers.append(
            nn.Sequential(
                VGGLayer(64, 128),
                VGGLayer(128, 128),
            )
        )
        layers.append(
            nn.Sequential(
                VGGLayer(128, 256),
                VGGLayer(256, 256),
                VGGLayer(256, 256),
            )
        )
        layers.append(
            nn.Sequential(
                VGGLayer(256, 512),
                VGGLayer(512, 512),
                VGGLayer(512, 512),
            )
        )

        self.dconvs = nn.ModuleList(layers)
        self.dlast_conv = nn.Sequential(
            nn.Conv2d(512, out_channels, 4, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.Tanh(),
        )
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skips = []
        h = x
        for dconv in self.dconvs:
            h = dconv(h)
            skips.append(h)

            # dim = dim / 2
            h = self.max_pool(h)

        return torch.flatten(self.dlast_conv(h), start_dim=1), skips


class Decoder(nn.Module):
    upfirst_conv: nn.Module
    upconvs: nn.ModuleList
    uplast_conv: nn.Module
    upsampling: nn.Module

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        upconvs = []
        upconvs.append(
            nn.Sequential(
                VGGLayer(512 * 2, 512), VGGLayer(512, 512), VGGLayer(512, 256)
            )
        )
        upconvs.append(
            nn.Sequential(
                VGGLayer(256 * 2, 256), VGGLayer(256, 256), VGGLayer(256, 128)
            )
        )
        upconvs.append(
            nn.Sequential(VGGLayer(128 * 2, 128), VGGLayer(128, 64))
        )

        self.upfirst_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.upconvs = nn.ModuleList(upconvs)
        self.uplast_conv = nn.Sequential(
            VGGLayer(64 * 2, 64),
            nn.ConvTranspose2d(64, 3, 3, 1, 1),
            nn.Sigmoid(),
        )
        self.upsampling = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]):
        # batch_size * in_channels * 1 * 1
        h = self.upfirst_conv(x.unsqueeze(-1).unsqueeze(-1))
        h = self.upsampling(h)

        number_of_upconvs = len(self.upconvs)
        for i, upconv in enumerate(self.upconvs):
            h = upconv(torch.concat([h, skips[number_of_upconvs - i]], dim=1))
            h = self.upsampling(h)

        return self.uplast_conv(torch.concat([h, skips[0]], dim=1))


class EmbeddedLSTM(nn.Module):
    embedding: nn.Module
    lstm: nn.ModuleList
    fcn: nn.Module

    _hiddens: List[torch.Tensor]

    def __init__(
        self,
        batch_size: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList(
            [nn.LSTMCell(hidden_size, hidden_size) for _ in range(num_layers)]
        )
        self.fcn = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.Tanh(),
        )

        self._hiddens = []
        for _ in range(num_layers):
            self._hiddens.append(torch.zeros(2, batch_size, hidden_size))

    def init_hiddens(self, device):
        for i in range(len(self._hiddens)):
            self._hiddens[i] = self._hiddens[i].to(device)
            self._hiddens[i].data.fill_(0)

    def forward(self, x: torch.Tensor):
        h_in = self.embedding(x)
        # TODO: Check if it is bug
        for i, layer in enumerate(self.lstm):
            h_0, c_0 = self._hiddens[i]
            h_in, c_n = torch.stack(layer(h_in, (h_0, c_0)), dim=0)
            self._hiddens[i] = torch.stack(
                [h_in.detach(), c_n.detach()], dim=0
            )

        return self.fcn(h_in)


class EmbeddedLSTMWithGaussian(nn.Module):
    embedding: nn.Module
    lstm: nn.ModuleList
    mu_fcn: nn.Module
    logvar_fcn: nn.Module

    _hiddens: nn.Parameter

    def __init__(
        self,
        batch_size: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList(
            [nn.LSTMCell(hidden_size, hidden_size) for _ in range(num_layers)]
        )
        self.mu_fcn = nn.Linear(hidden_size, output_size)
        self.logvar_fcn = nn.Linear(hidden_size, output_size)

        self._hiddens = []
        for _ in range(num_layers):
            self._hiddens.append(torch.zeros(2, batch_size, hidden_size))

    def init_hiddens(self, device):
        for i in range(len(self._hiddens)):
            self._hiddens[i] = self._hiddens[i].to(device)
            self._hiddens[i].data.fill_(0)

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        h_in = self.embedding(x)
        for i, layer in enumerate(self.lstm):
            h_0, c_0 = self._hiddens[i]
            h_in, c_n = layer(h_in, (h_0, c_0))
            self._hiddens[i] = torch.stack(
                [h_in.detach(), c_n.detach()], dim=0
            )

        mu = self.mu_fcn(h_in)
        logvar = self.logvar_fcn(h_in)
        z = self._reparameterize(mu, logvar)
        return z, mu, logvar


class CVAE(nn.Module):
    condition_embedding: nn.Linear
    encoder: Encoder
    predictor: EmbeddedLSTM
    posterior: EmbeddedLSTMWithGaussian
    decoder: Decoder

    _encoder_skips: List[torch.Tensor]
    _z_dim: int

    def __init__(
        self,
        batch_size: int,
        condition_embedding_input_features: int,
        condition_embedding_output_features: int,
        encoder_input_channels: int,
        encoder_output_channels: int,
        rnn_hidden_size: int,
        predictor_rnn_layers: int,
        posterior_rnn_output_size: int,
        posterior_rnn_layers: int,
    ) -> None:
        super().__init__()
        self.condition_embedding = nn.Linear(
            in_features=condition_embedding_input_features,
            out_features=condition_embedding_output_features,
            bias=False,
        )
        self.encoder = Encoder(encoder_input_channels, encoder_output_channels)
        self.predictor = EmbeddedLSTM(
            batch_size=batch_size,
            input_size=(
                condition_embedding_output_features
                + encoder_output_channels
                + posterior_rnn_output_size
            ),
            hidden_size=rnn_hidden_size,
            output_size=encoder_output_channels,
            num_layers=predictor_rnn_layers,
        )
        self.posterior = EmbeddedLSTMWithGaussian(
            batch_size=batch_size,
            input_size=encoder_output_channels,
            hidden_size=rnn_hidden_size,
            output_size=posterior_rnn_output_size,
            num_layers=posterior_rnn_layers,
        )
        self.decoder = Decoder(in_channels=encoder_output_channels)

        self._encoder_skips = []
        self._z_dim = posterior_rnn_output_size

    def init_hiddens(self, device: str):
        self.predictor.init_hiddens(device)
        self.posterior.init_hiddens(device)

    def predict(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        update_skips: bool = True,
    ) -> torch.Tensor:
        condition = self.condition_embedding(condition)
        # batch_size, channels, x, x
        if update_skips:
            h, self._encoder_skips = self.encoder(x)
        else:
            h, _ = self.encoder(x)

        # sample from N(0, 1)
        z = torch.randn(h.size(0), self._z_dim, device=h.device)

        # frame predictor
        # batch_size, input_size
        g = self.predictor(torch.concat([condition, h, z], dim=1))
        # batch_size, output_size
        x_pred = self.decoder(g, self._encoder_skips)

        return x_pred

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        target: torch.Tensor,
        update_skips: bool = True,
    ):
        condition = self.condition_embedding(condition)
        # batch_size, channels, x, x
        if update_skips:
            h, self._encoder_skips = self.encoder(x)
        else:
            h, _ = self.encoder(x)

        h_target, _ = self.encoder(target)

        # inference network
        # batch_size, input_size
        z, mu, logvar = self.posterior(h_target)

        # frame predictor
        # batch_size, input_size
        g = self.predictor(torch.concat([condition, h, z], dim=1))
        # batch_size, output_size
        x_pred = self.decoder(g, self._encoder_skips)

        return x_pred, mu, logvar
