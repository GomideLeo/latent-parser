import torch
import torch.nn.functional as F

class Laplacian(object):

    def __init__(self, inplace=False, kernel='negative'):
        super(Laplacian, self)
        self.inplace = inplace
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

        return conv if self.inplace else torch.cat((image, conv))