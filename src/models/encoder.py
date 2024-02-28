import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce

class Encoder(nn.Module):
    def __init__(
        self,
        output_dim=32,
        n_channels=1,
        use_batchnorm=False,
        use_dropout=False,
        conv_layers=[(5, 32), (3, 64), (3, 128)],
        conv_pooling=[2, 2, 2],
        linear_input=(128, 14, 14),
        linear_layers=[256, 128],
        device=None,
    ):
        super(Encoder, self).__init__()

        self.kwargs = {
            'output_dim': output_dim,
            'n_channels': n_channels,
            'use_batchnorm': use_batchnorm,
            'use_dropout': use_dropout,
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
        self.use_dropout = use_dropout
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

        self.distribution_mean = nn.Linear(self.linear_layers[-1], output_dim)
        self.distribution_variance = nn.Linear(self.linear_layers[-1], output_dim)
    
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

            if self.use_dropout:
                linear_layers.append(nn.Dropout(0.15))

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
                conv_layers.append(nn.Conv2d(self.n_channels, l[1], kernel_size=l[0]))
            else:
                conv_layers.append(
                    nn.Conv2d(self.conv_layers[i - 1][1], l[1], kernel_size=l[0])
                )

            if self.use_batchnorm:
                conv_layers.append(nn.BatchNorm2d(self.channels[i]))

            # Here we use GELU as activation function
            conv_layers.append(nn.GELU())

            if self.use_dropout:
                conv_layers.append(nn.Dropout2d(0.15))

            if p is not False:
                conv_layers.append(nn.MaxPool2d(p))

        return conv_layers

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.linear(x)

        mean, var = self.distribution_mean(x), self.distribution_variance(x)
        x = self.sample_latent_features([mean, var])

        return x, mean, var

    def sample_latent_features(self, distribution):
        # if not self.training: return distribution[0]
        distribution_mean, distribution_variance = distribution
        shape = distribution_variance.shape
        batch_size = shape[0]
        random = torch.normal(mean=0, std=1, size=(batch_size, shape[1])).to(
            self.device
        )

        return distribution_mean + torch.exp(0.5 * distribution_variance) * random
