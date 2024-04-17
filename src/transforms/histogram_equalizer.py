from functools import reduce
import torch
import numpy as np

class HistogramEqualizer(object):

    def __init__(self, L=256, min_value=0, max_value=1):
        self.min = min_value
        self.max = max_value
        self.L = L

    def __call__(self, sample):
        image = sample

        if len(image.shape) == 3:
            for (i, dim) in enumerate(image):
                total = reduce(lambda a, b: a*b, dim.shape)

                unique_vals = torch.unique(torch.flatten(dim))
                hist = torch.histogram(torch.flatten(dim), bins=torch.tensor([*unique_vals, unique_vals.max()+1]))
                cdf = {v.item(): n.item() for v, n in zip(unique_vals, torch.cumsum(hist.hist, 0))}
                const = ((self.L) / total)

                get_val = np.vectorize(lambda v: int(cdf[v] * const))

                image[i] = torch.tensor(get_val(dim)) / self.L
                image[i] = image[i] * (self.max - self.min) + self.min
        else:
            total = reduce(lambda a, b: a*b, image.shape)

            unique_vals = torch.unique(torch.flatten(image))
            hist = torch.histogram(torch.flatten(image), bins=torch.tensor([*unique_vals, unique_vals.max()+1]))
            cdf = {v.item(): n.item() for v, n in zip(unique_vals, torch.cumsum(hist.hist, 0))}
            const = ((self.L-1) / total)

            get_val = np.vectorize(lambda v: int(cdf[v.item()] * const))

            image = get_val(image) / (self.L-1)
            image = image * (self.max - self.min) + self.min

        return image
