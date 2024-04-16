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

        if len(image.shape) == 3:
            for (i, dim) in enumerate(image):
                min_value = torch.min(dim)
                max_value = torch.max(dim)

                image[i] = (dim - min_value) / (max_value - min_value)
                image[i] = dim * (self.max - self.min) + self.min
        else:
            min_value = torch.min(image)
            max_value = torch.max(image)

            image = (image - min_value) / (max_value - min_value)
            image = image * (self.max - self.min) + self.min

        return image
