import torch
import torch.nn as nn
from functools import reduce
from . import Downsample, ResidualBlock

class ResidualEncoder(nn.Module):
    def __init__(
        self,
        output_dim=32,
        n_channels=1,
        use_batchnorm=False,
        dropout=0,
        conv_layers=[32, 64, 128],
        conv_pooling=[2, 2, 2],
        linear_input=(128, 16, 16),
        linear_layers=[256, 128],
        variational=True,
        device=None,
    ):
        super(ResidualEncoder, self).__init__()

        self.kwargs = {
            'output_dim': output_dim,
            'n_channels': n_channels,
            'use_batchnorm': use_batchnorm,
            'dropout': dropout,
            'conv_layers': conv_layers,
            'conv_pooling': conv_pooling,
            'linear_input': linear_input,
            'linear_layers': linear_layers,
            'device': device,
        }

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # bottleneck dimentionality
        self.output_dim = output_dim

        self.n_channels = n_channels

        # variables deciding if using dropout and batchnorm in model
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm

        # convolutional layer hyper parameters
        self.conv_layers = conv_layers
        self.conv_pooling = conv_pooling
        if len(conv_layers) != len(conv_pooling):
            raise Exception("conv_layers and conv_pooling should have the same length")
        self.conv = self._get_convs()

        # layers for latent space projection
        self.flatten = nn.Flatten(1)
        self.fc_dim = reduce(lambda a, b: a*b, linear_input)
        self.linear_layers = linear_layers
        self.linear = self._get_linear()

        self.variational = variational

        if variational:
            self.distribution_mean = nn.Linear(self.linear_layers[-1], output_dim)
            self.distribution_variance = nn.Linear(self.linear_layers[-1], output_dim)
        else:
            self.linear.append(nn.Linear(self.linear_layers[-1], output_dim))
            self.linear.append(nn.GELU())

            if self.dropout:
                self.linear.append(nn.Dropout(0.15))

    
    def _get_linear(self):
        """
        generating linear layers based on model's
        hyper parameters
        """
        linear_layers = nn.Sequential()
        for i, l in enumerate(self.linear_layers):
            # The input channel of the first layer is fc_dim
            if i == 0:
                linear_layers.append(nn.Linear(self.fc_dim, l))
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
        for i, (l, p) in enumerate(zip(self.conv_layers, self.conv_pooling)):
            # The input channel of the first layer is 1
            if i == 0:
                conv_layers.append(ResidualBlock(self.n_channels, l, self.dropout, self.use_batchnorm))
            else:
                conv_layers.append(
                    ResidualBlock(self.conv_layers[i - 1], l, self.dropout, self.use_batchnorm)
                )

            # Here we use SiLU as activation function
            conv_layers.append(nn.SiLU())

            if p is not False:
                conv_layers.append(Downsample(l))
                # conv_layers.append(nn.MaxPool2d(p))

        return conv_layers

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.linear(x)

        if self.variational:
            mean, var = self.distribution_mean(x), self.distribution_variance(x)
            x = self.sample_latent_features([mean, var])
            return x, mean, var

        return x

    def sample_latent_features(self, distribution):
        # if not self.training: return distribution[0]
        distribution_mean, distribution_variance = distribution
        shape = distribution_variance.shape
        batch_size = shape[0]
        random = torch.normal(mean=0, std=1, size=(batch_size, shape[1])).to(
            self.device
        )

        return distribution_mean + torch.exp(0.5 * distribution_variance) * random
