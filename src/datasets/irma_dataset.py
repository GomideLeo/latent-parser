import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from skimage import io
from sklearn.preprocessing import LabelEncoder

class IrmaDataset(Dataset):
    irma_classmap = LabelEncoder().fit(['BI-RADS A', 'BI-RADS B', 'BI-RADS C', 'BI-RADS D'])

    def __init__(self, metadata_file='featureS.txt', root_dir='./datasets/IRMA/', transform=None):
        """
        Arguments:
            csv_file (string): Path to the metadata file.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.metadata_frame = self._read_metadata(root_dir, metadata_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.metadata_frame.iloc[idx, 0] + '.png')
        image = io.imread(img_name)
        label = int(self.metadata_frame.iloc[idx, 1])
        # label = np.array(label, dtype=float)

        if self.transform:
            image = self.transform(image)

        return image, label


    def _read_metadata(self, root_dir, metadata_file):
        files_metadatas = []

        with open(root_dir+metadata_file, 'r') as paths:
            next_path = paths.readline()

            while next_path:
                label = paths.readline()
                next_path, label = next_path[:-1], label[:-1] # remove '\n'
                files_metadatas.append((next_path, label))

                next_path = paths.readline()

        return pd.DataFrame(files_metadatas, columns=['file_name', 'label'])

    @classmethod
    def get_class_label(cls, labels):
        return cls.irma_classmap.inverse_transform(labels)

    @classmethod
    def get_class_label_value(cls, labels):
        return cls.irma_classmap.transform(labels)