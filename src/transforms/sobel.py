import torch
import torch.nn.functional as F

class Sobel(object):

    def __init__(self, clamp=(0, 1)):
        super(Sobel, self)
        self.clamp=clamp

    def __call__(self, sample):
        image = sample
        s_a = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]).view(
            (1, 1, 3, 3)
        )
        s_b = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]).view(
            (1, 1, 3, 3)
        )

        conv_a = F.conv2d(image, s_a, padding=1)
        conv_b = F.conv2d(image, s_b, padding=1)

        if self.clamp != False and type(self.clamp) == tuple:
            val = torch.cat((image, torch.clamp(torch.sqrt(torch.pow(conv_a, 2) + torch.pow(conv_b, 2)), min=self.clamp[0], max=self.clamp[1])))
        else:
            val = torch.cat((image, torch.sqrt(torch.pow(conv_a, 2) + torch.pow(conv_b, 2))))

        return val