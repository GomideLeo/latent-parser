import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from skimage import io
from sklearn.preprocessing import LabelEncoder

class IrmaDataset(Dataset):
    irma_classmap = LabelEncoder().fit(['BI-RADS I', 'BI-RADS II', 'BI-RADS III', 'BI-RADS IV'])

    def __init__(self, metadata_file='featureS.txt', root_dir='./datasets/IRMA/', transform=None, return_images=False, return_path=False):
        """
        Arguments:
            csv_file (string): Path to the metadata file.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.return_images = return_images
        self.root_dir = root_dir
        self.transform = transform
        self.metadata_frame = self._read_metadata(root_dir, metadata_file)
        self.return_path = return_path

    def __len__(self):
        return len(self.metadata_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = self.metadata_frame.iloc[idx, -1]
        path = self.metadata_frame.iloc[idx, 0]

        if self.return_images:
            image = self.metadata_frame.iloc[idx, 1]
        else:
            img_name = os.path.join(self.root_dir,
                                    path + '.png')
            image = io.imread(img_name)
            # label = np.array(label, dtype=float)

            if self.transform:
                image = self.transform(image)

        if self.return_path:
            return image, label, path
        else:
            return image, label


    def _read_metadata(self, root_dir, metadata_file):
        files_metadatas = []

        with open(root_dir+metadata_file, 'r') as paths:
            next_path = paths.readline()

            while next_path:
                label = paths.readline()
                next_path, label = next_path[:-1], int(label[:-1]) # remove '\n'

                if self.return_images:
                    img_name = os.path.join(self.root_dir, next_path + '.png')
                    image = io.imread(img_name)

                    if self.transform:
                        image = self.transform(image)
                    
                    files_metadatas.append((next_path, image, label))
                else:
                    files_metadatas.append((next_path, label))

                next_path = paths.readline()

        return pd.DataFrame(files_metadatas, columns=['file_name', 'image', 'label'] if self.return_images else ['file_name', 'label'])

    @classmethod
    def get_class_label(cls, labels):
        return cls.irma_classmap.inverse_transform(labels)

    @classmethod
    def get_class_label_value(cls, labels):
        return cls.irma_classmap.transform(labels)