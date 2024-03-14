import torch
import torch.nn as nn
from .modules import Encoder, Decoder

class PVAE(nn.Module):

    def __init__(
        self,
        latent_dim=32,
        n_channels=1,
        use_batchnorm=False,
        use_dropout=False,
        conv_layers=[(5, 32), (3, 64), (3, 128)],
        conv_pooling=[2, 2, 2],
        linear_input=(128, 14, 14),
        linear_layers=[256, 128],
        n_classes=2,
        pred_layers=[],
        clamp_output=(0, 1),
        device=None,
    ):
        super(PVAE, self).__init__()

        self.kwargs = {
            'latent_dim': latent_dim,
            'n_channels': n_channels,
            'use_batchnorm': use_batchnorm,
            'use_dropout': use_dropout,
            'conv_layers': conv_layers,
            'conv_pooling': conv_pooling,
            'linear_input': linear_input,
            'linear_layers': linear_layers,
            'n_classes': n_classes,
            'pred_layers': pred_layers,
            'device': device,
        }

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.n_classes = n_classes
        self.pred_layers = pred_layers
        self.latent_dim = latent_dim
        self.n_channels = n_channels
        self.use_dropout = use_dropout

        self.encoder = Encoder(
            output_dim=latent_dim,
            n_channels=n_channels,
            use_batchnorm=use_batchnorm,
            use_dropout=use_dropout,
            conv_layers=conv_layers,
            conv_pooling=conv_pooling,
            linear_input=linear_input,
            linear_layers=linear_layers,
            device=device,
        )
        self.predictor = self._get_pred()
        self.decoder = Decoder(
            input_dim=latent_dim,
            n_channels=n_channels,
            use_batchnorm=use_batchnorm,
            use_dropout=use_dropout,
            conv_layers=conv_layers[::-1],
            conv_upsample=conv_pooling[::-1],
            linear_output=linear_input,
            linear_layers=linear_layers[::-1],
            clamp_output=clamp_output,
            device=device,
        )

    def _get_pred(self):
        pred_layers = nn.Sequential()
        prev_layer = self.latent_dim
        for i, l in enumerate(self.pred_layers):
            # The input channel of the first layer is fc_dim
            pred_layers.append(nn.Linear(prev_layer, l))
            prev_layer = l

            # Here we use GELU as activation function
            pred_layers.append(nn.GELU())

            if self.use_dropout:
                pred_layers.append(nn.Dropout(0.15))

        if self.n_classes == 2:
            pred_layers.append(nn.Linear(prev_layer, 1))
            pred_layers.append(nn.Sigmoid())
        else:
            pred_layers.append(nn.Linear(prev_layer, self.n_classes))
            pred_layers.append(nn.Softmax(-1))

        return pred_layers

    def forward(self, x):
        latent, mean, var = self.encoder(x)
        return self.decoder(latent), self.predictor(latent), latent, mean, var
