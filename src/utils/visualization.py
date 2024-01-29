import torch, torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def sample_images(data_source, n=9, figsize=(5, 5)):
    """Shows a grid of n images from a PyTorch dataset with labels as titles.

    Args:
        dataset: A Torch Dataset containing images and labels.
        n: The number of images to show (default: 9).
    """

    if isinstance(data_source, DataLoader):
        images, labels = [], []
        i = 0
        for img, label in data_source:
            for im, l in zip(img, label):
                images.append(im)
                labels.append(l)
                i+=1

                if i > n - 1:
                    break
            if i > n - 1:
                    break
    else:
        images, labels = zip(*[data_source[i] for i in range(n)])


    images = torch.stack(images)  # Create a grid of images
    images = images.permute(0, 2, 3, 1)  # Reshape for matplotlib (C, H, W -> H, W, C)

    grid_img = torchvision.utils.make_grid(images, nrow=3)
    fig, axes = plt.subplots(3, 3, figsize=figsize)  # Create subplots for individual images

    for i in range(n):
        if images[i].shape[-1] == 1:  # Grayscale image
            axes[i // 3, i % 3].imshow(images[i], cmap='gray')  # Apply cmap
        else:
            axes[i // 3, i % 3].imshow(images[i])  # Color image

        axes[i // 3, i % 3].axis('off')
        axes[i // 3, i % 3].set_title(str(labels[i].item()))  # Set label as title for each image

    fig.suptitle(f"Grid of {n} Images with Labels", fontsize=16)  # Add overall title
    plt.tight_layout()
    plt.show()