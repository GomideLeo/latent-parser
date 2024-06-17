import torch
import torch.nn.functional as F

class Laplacian(object):

    def __init__(self, mode=None, kernel='negative', clamp=False):
        super(Laplacian, self)
        self.inplace = True if mode == 'inplace' else False
        self.add = True if mode == 'add' else False
        self.clamp = clamp
        if type(kernel) == torch.Tensor:
            self.kernel = kernel
        elif kernel == 'negative':
            self.kernel = torch.tensor([[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]]).view(
                (1, 1, 3, 3)
            )
        elif kernel == 'positive':
            self.kernel = self.kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(
                (1, 1, 3, 3)
            )
        else:
            raise Exception('kernel should be a Tensor, "postive" or "negative"')

    def __call__(self, sample):
        image = sample

        conv = F.conv2d(image, self.kernel, padding='same')

        if self.add: 
            image = image + conv

            if self.clamp != False and type(self.clamp) == tuple:
                image = torch.clamp(image, self.clamp[0], self.clamp[1])

            return image
        else:
            if self.clamp != False and type(self.clamp) == tuple:
                conv = torch.clamp(conv, self.clamp[0], self.clamp[1])

            return conv if self.inplace else torch.cat((image, conv))