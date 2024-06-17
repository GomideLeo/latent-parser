from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import numpy as np
import torch


def train_val_split(dataset, val_split=0.25, random_state=None):
    train_idx, val_idx = train_test_split(
        list(range(len(dataset))), test_size=val_split, random_state=random_state
    )
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets


def get_latent(model, data, label_mapper=lambda l: l, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = device

    model.to(device)
    latent_features = (np.ndarray((0, model.latent_dim)), np.ndarray(0))

    with torch.no_grad():
        data_len = 0
        for i, data in enumerate(data, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data[0].to(device)
            data_len += len(data[0])
            labels = data[1]

            outputs = model.encoder(inputs)
            latent = outputs[0] if type(outputs) == tuple else outputs

            latent_features = (
                np.concatenate((latent_features[0], latent.cpu().detach().numpy())),
                np.concatenate(
                    (latent_features[1], label_mapper(labels.detach().numpy()))
                ),
            )

    return latent_features

def predict(model, data, return_class=False, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = device

    model.to(device)
    result = None

    with torch.no_grad():
        data_len = 0
        for i, data in enumerate(data, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data[0].to(device)
            data_len += len(data[0])
            labels = data[1].numpy()

            outputs = model(inputs)
            outputs = outputs.cpu().detach().numpy()
            if return_class:
                outputs = np.argmax(outputs, axis=1)

            if result == None:
                result = (outputs, labels)
            else:
                result = (
                    np.concatenate((result[0], outputs)),
                    np.concatenate((result[1], labels)),
                )

    return result
