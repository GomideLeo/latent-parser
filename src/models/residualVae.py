import torch
import torch.nn as nn
from .modules import ResidualDecoder, ResidualEncoder

class ResidualVAE(nn.Module):

    def __init__(
        self,
        latent_dim=32,
        n_channels=1,
        use_batchnorm=False,
        dropout=0,
        conv_layers=[(5, 32), (3, 64), (3, 128)],
        conv_pooling=[2, 2, 2],
        linear_input=(128, 14, 14),
        linear_layers=[256, 128],
        clamp_output=(0, 1),
        device=None,
    ):
        super(ResidualVAE, self).__init__()

        self.kwargs = {
            'latent_dim': latent_dim,
            'n_channels': n_channels,
            'use_batchnorm': use_batchnorm,
            'dropout': dropout,
            'conv_layers': conv_layers,
            'conv_pooling': conv_pooling,
            'linear_input': linear_input,
            'linear_layers': linear_layers,
            'clamp_output': clamp_output,
            'device': device,
        }

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.latent_dim = latent_dim
        self.n_channels = n_channels

        self.encoder = ResidualEncoder(
            output_dim=latent_dim,
            n_channels=n_channels,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
            conv_layers=conv_layers,
            conv_pooling=conv_pooling,
            linear_input=linear_input,
            linear_layers=linear_layers,
            device=device,
            variational=True
        )
        self.decoder = ResidualDecoder(
            input_dim=latent_dim,
            n_channels=n_channels,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
            conv_layers=conv_layers[::-1],
            conv_upsample=conv_pooling[::-1],
            linear_output=linear_input,
            linear_layers=linear_layers[::-1],
            clamp_output=clamp_output,
            device=device,
        )

    def forward(self, x):
        latent, mean, var = self.encoder(x)
        return self.decoder(latent), latent, mean, var
