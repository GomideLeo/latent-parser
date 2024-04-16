import torch
import torch.nn.functional as F

class GammaCorrection(object):

    def __init__(self, gamma=1, dim=None):
        super(GammaCorrection, self)
        self.gamma = gamma
        self.dim = dim

    def __call__(self, sample):
        image = sample

        if self.dim is None:
            image = torch.pow(image, self.gamma)
        else:
            image[self.dim] = torch.pow(image[self.dim], self.gamma) 

        return image