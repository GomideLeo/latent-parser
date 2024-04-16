import torch

class HistogramNormalizer(object):

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
            histogram = torch.unique(image, sorted=True, return_counts=True)

            image = (image - min_value) / (max_value - min_value)
            image = image * (self.max - self.min) + self.min

        return image
