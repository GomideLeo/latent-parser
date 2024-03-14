import torch
import torch.nn as nn
import torch.functional as F
from ..losses.kl_loss import get_kl_loss
from ..losses.reconstruction_loss import get_reconstruction_loss

def get_metrics_str(metrics, prefix=""):
    return " - ".join([f"{prefix}{l[0]}: {l[1]:.4f}" for l in metrics.items()])

def kl_reconstruction_loss(input, model_out, y_true):
    out, latent, mean, var = model_out

    kl_loss = get_kl_loss(mean, var)
    rec_loss = get_reconstruction_loss(input, out)

    loss = kl_loss + rec_loss

    return loss, dict(rec_loss=rec_loss, kl_loss=kl_loss)

def pred_kl_reconstruction_loss(input, model_out, y_true, criterion=nn.CrossEntropyLoss(reduction='sum')):
    out, pred, latent, mean, var = model_out

    kl_loss = get_kl_loss(mean, var)
    rec_loss = get_reconstruction_loss(input, out)
    pred_loss = criterion(pred, y_true)

    _, predicted = torch.max(pred, 1)
    accuracy = (predicted == y_true).sum()

    loss = kl_loss + rec_loss + pred_loss

    return loss, dict(rec_loss=rec_loss, kl_loss=kl_loss, pred_loss=pred_loss, accuracy=accuracy)

def pred_loss(input, model_out, y_true, criterion=nn.CrossEntropyLoss(reduction='sum')):
    pred = model_out

    predict_loss = criterion(pred, y_true)

    _, predicted = torch.max(pred, 1)
    accuracy = (predicted == y_true).sum()

    loss = predict_loss

    return loss, dict(predict_loss=predict_loss, accuracy=accuracy)

def validate(model, data, loss_func=None, device='cuda'):
    running_metrics = {'loss': 0.0}

    with torch.no_grad():
        data_len = 0
        for i, (d, l) in enumerate(data, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = d.to(device)
            data_len += len(d)
            labels = l.to(device)

            outputs = model(inputs)

            loss, metrics = loss_func(inputs, outputs, labels)

            running_metrics['loss'] += loss.item()

            for metric in metrics.items():
                running_metrics.setdefault(metric[0], 0)
                running_metrics[metric[0]] += metric[1].item()

    return {k: v/data_len for k, v in running_metrics.items()}

def train(model, optimizer, train_data, val_data, epochs, loss_func, device='cuda'):
    hist_metrics = []

    for epoch in range(epochs):  # loop over the dataset multiple times

        train_metrics = {'loss': 0.0}
        train_len = 0

        for i, data in enumerate(train_data, 0):
            inputs = data[0].to(device)
            train_len += len(data[0])
            labels = data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss, metrics = loss_func(inputs, outputs, labels)

            loss.backward()

            train_metrics['loss'] += loss.item()
            for metric in metrics.items():
                train_metrics.setdefault(metric[0], 0.0)
                train_metrics[metric[0]] += metric[1].item()

            optimizer.step()

        
        val_metrics = validate(model, val_data, loss_func)

        train_metrics = {k: v/train_len for k, v in train_metrics.items()}

        hist_metrics.append((train_metrics, val_metrics))
        print(f'[epoch: {epoch+1}] {get_metrics_str(train_metrics, "train_")} - {get_metrics_str(val_metrics, "val_")}')

    print('Finished Training')
    return hist_metrics