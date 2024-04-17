import torch.nn.functional as F

def get_reconstruction_loss(y_true, y_pred, device='cuda'):
    reconstruction_loss = F.mse_loss(y_pred, y_true, reduction='sum').to(device)
    reconstruction_loss = reconstruction_loss/(8 * y_pred.shape[0] * y_pred.shape[1])

    return reconstruction_loss