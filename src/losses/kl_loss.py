import torch

def get_kl_loss(mean, logvar):
    kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

    return kl_loss