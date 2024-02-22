import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):

    def __init__(
        self,
        input_dim=32,
        use_batchnorm=False,
        use_dropout=False,
        n_channels=1,
        conv_layers=[(3, 32), (3, 64), (3, 128)],
        conv_pooling=[2, 2, 2],
        linear_output=(128 * 14 * 14),
        linear_layers=[256, 128],
        device=None,
    ):

        super(Decoder, self).__init__()

        # variables deciding if using dropout and batchnorm in model
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm

        self.fc_dim = linear_output
        self.input_dim = input_dim

        # Conv layer hypyer parameters
        self.layers = LAYERS
        self.kernels = KERNELS
        self.channels = CHANNELS[::-1]  # flip the channel dimensions
        self.strides = STRIDES

        # In decoder, we first do fc project, then conv layers
        self.linear = nn.Linear(self.input_dim, self.fc_dim)
        self.conv = self._get_convs()

        self.output = nn.Conv2d(self.channels[-1], 1, kernel_size=1, stride=1)

    def _get_convs(self):
        conv_layers = nn.Sequential()
        for i in range(self.layers):

            if i == 0:
                conv_layers.append(
                    nn.ConvTranspose2d(
                        self.channels[i],
                        self.channels[i],
                        kernel_size=self.kernels[i],
                        stride=self.strides[i],
                        padding=1,
                        output_padding=1 if self.strides[i] == 2 else 0,
                    )
                )

            else:
                conv_layers.append(
                    nn.ConvTranspose2d(
                        self.channels[i - 1],
                        self.channels[i],
                        kernel_size=self.kernels[i],
                        stride=self.strides[i],
                        padding=1,
                        output_padding=1 if self.strides[i] == 2 else 0,
                    )
                )

            if self.use_batchnorm and i != self.layers - 1:
                conv_layers.append(nn.BatchNorm2d(self.channels[i]))

            conv_layers.append(nn.GELU())

            if self.use_dropout:
                conv_layers.append(nn.Dropout2d(0.15))

        return conv_layers

    def forward(self, x):
        x = self.linear(x)
        # reshape 3D tensor to 4D tensor
        x = x.reshape(x.shape[0], 512, 4, 4)
        x = self.conv(x)
        return self.output(x)
