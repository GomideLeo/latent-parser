import torch, torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

def sample_images(data_source, n=9, figsize=(5, 5), label_mapper=lambda l: l):
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

    labels = label_mapper(labels)

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

def plot_reconstructions(model, data, inv_normalize=None, n=4, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = device

    fig, axs = plt.subplots(2, n)

    with torch.no_grad():
        for i, data in enumerate(data, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            outputs = model(inputs)

            for ax, reconstructed, original in zip(range(n), outputs[0], inputs):
                if inv_normalize is not None:
                    reconstructed = inv_normalize(reconstructed)      # unnormalize
                    original = inv_normalize(original)                # unnormalize

                axs[0][ax].imshow(reconstructed[0].cpu(), cmap='gray')
                # axs[0][ax].imshow(reconstructed.cpu().permute(1,2,0))
                axs[1][ax].imshow(original[0].cpu(), cmap='gray')
                # axs[1][ax].imshow(original.cpu().permute(1,2,0))
                axs[0][ax].axis('off')
                axs[1][ax].axis('off')
            plt.show()

            break

def plot_2d(pos, labels, dataset, cmap='tab10'):
    cmap = plt.colormaps[cmap]
    plt.figure(figsize=(8, 6))
    for c, cl in zip(np.unique(labels), pd.Series(dataset.get_class_label_value(np.unique(labels))).map(cmap)):
        idxs = np.argwhere(labels == c)
        plt.scatter(
            pos[idxs, 0],
            pos[idxs, 1],
            color=cl,
            label=c
        )
    plt.legend()
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()

def plot_history(history, show=True):
    hist_df = None
    
    train_hist_df = pd.DataFrame(map(lambda h: h[0], history))
    val_hist_df = pd.DataFrame(map(lambda h: h[1], history))
    # hist_df.columns=["loss", "accuracy", "val_loss", "val_accuracy"]
    train_hist_df.index = np.arange(1, len(train_hist_df) + 1)
    val_hist_df.index = np.arange(1, len(val_hist_df) + 1)

    fig, axs = plt.subplots(nrows=len(train_hist_df.columns), sharex=True, figsize=(12, 8))
    for i, c in enumerate(train_hist_df.columns):
        axs[i].plot(val_hist_df[c], lw=3, label=f'Validation {c.replace("_", " ")}')
        axs[i].plot(train_hist_df[c], lw=3, label=f'Training {c.replace("_", " ")}')
        axs[i].set_ylabel(c.replace('_', ' ').capitalize())
        axs[i].set_xlabel('Epoch')
        axs[i].grid()
        axs[i].legend(loc=0)

    if show:
        plt.show()
    else:
        return fig, axs