import torch.nn.functional as F

def get_reconstruction_loss(y_true, y_pred, device='cuda'):
    reconstruction_loss = F.mse_loss(y_pred, y_true, reduction='sum').to(device)
    reconstruction_loss = reconstruction_loss/y_pred.shape[0]

    return reconstruction_loss