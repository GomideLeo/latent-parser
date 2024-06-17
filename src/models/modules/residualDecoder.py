import torch
import torch.nn as nn
from functools import reduce
from . import Upsample, ResidualBlock

class ResidualDecoder(nn.Module):
    def __init__(
        self,
        input_dim=32,
        use_batchnorm=False,
        dropout=0,
        n_channels=1,
        conv_layers=[128, 64, 32],
        conv_upsample=[2, 2, 2],
        linear_output=(128, 16, 16),
        linear_layers=[128, 256],
        clamp_output=(0,1),
        device=None,
    ):
        super(ResidualDecoder, self).__init__()

        self.kwargs = {
            'input_dim': input_dim,
            'use_batchnorm': use_batchnorm,
            'dropout': dropout,
            'n_channels': n_channels,
            'conv_layers': conv_layers,
            'conv_upsample': conv_upsample,
            'linear_output': linear_output,
            'linear_layers': linear_layers,
            'clamp_output': clamp_output,
            'device': device,
        }

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # variables deciding if using dropout and batchnorm in model
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm

        self.n_channels = n_channels

        # In decoder, we first do fc project, then conv layers
        self.linear_output = linear_output
        self.fc_dim = reduce(lambda a, b: a*b, linear_output)
        self.input_dim = input_dim
        self.linear_layers = linear_layers
        self.linear = self._get_linear()

        self.unflatten = nn.Unflatten(1, (self.linear_layers[-1], 1, 1))

        # Conv layer hypyer parameters
        self.conv_layers = conv_layers
        self.conv_upsample = conv_upsample
        if len(conv_layers) != len(conv_upsample):
            raise Exception("conv_layers and conv_upsample should have the same length")
        self.conv = self._get_convs()

        self.output = nn.Sequential()
        self.output.append(nn.Conv2d(self.conv_layers[-1], self.n_channels, kernel_size=1, stride=1))
        if clamp_output is not None and clamp_output is not False:
            self.output.append(nn.Hardtanh(*clamp_output)) # clamp output between 0 and 1

    def _get_linear(self):
        """
        generating linear layers based on model's
        hyper parameters
        """
        linear_layers = nn.Sequential()
        for i, l in enumerate(self.linear_layers):
            # The input channel of the first layer is fc_dim
            if i == 0:
                linear_layers.append(nn.Linear(self.input_dim, l))
            else:
                linear_layers.append(nn.Linear(self.linear_layers[i - 1], l))

            # Here we use GELU as activation function
            linear_layers.append(nn.GELU())

            if self.dropout:
                linear_layers.append(nn.Dropout(self.dropout))

        return linear_layers
    
    def _get_convs(self):
        """
        generating convolutional layers based on model's
        hyper parameters
        """
        conv_layers = nn.Sequential()

        conv_layers.append(nn.ConvTranspose2d(self.linear_layers[-1], self.linear_output[0], kernel_size=self.linear_output[1:]))

        for i, (l, u) in enumerate(zip(self.conv_layers, self.conv_upsample)):
            if u is not False:
                if i == 0:
                    conv_layers.append(Upsample(self.linear_output[0]))
                else:
                    conv_layers.append(Upsample(self.conv_layers[i - 1]))

            # The input channel of the first layer is 1
            if i == 0:
                conv_layers.append(ResidualBlock(self.linear_output[0], l, self.dropout, self.use_batchnorm))
            else:
                conv_layers.append(
                    ResidualBlock(self.conv_layers[i - 1], l, self.dropout, self.use_batchnorm)
                )

            # Here we use GELU as activation function
            conv_layers.append(nn.SiLU())

        return conv_layers

    def forward(self, x):
        x = self.linear(x)
        x = self.unflatten(x)
        # reshape 3D tensor to 4D tensor
        # x = x.reshape(x.shape[0], 512, 4, 4)
        x = self.conv(x)
        return self.output(x)
        # return x
