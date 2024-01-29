import torch

class MinMaxScaler(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, min=0, max=1):
        self.min = min
        self.max = max

    def __call__(self, sample):
        image = sample

        min = torch.min(image)
        max = torch.max(image)

        img = (image - min) / (max - min)
        img = img * (self.max - self.min) + self.min

        return img
