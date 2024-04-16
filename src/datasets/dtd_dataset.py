import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from skimage import io
from sklearn.preprocessing import LabelEncoder
import glob as glob

class DTDDataset(Dataset):
    def __init__(self, root_dir='./datasets/DTD/dtd/images/*', transform=None, return_images=False, include_classes=None, exclude_classes=None):
        """
        Arguments:
            csv_file (string): Path to the metadata file.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        self.metadata_frame = self._read_metadata(root_dir, return_images, include_classes, exclude_classes)
        self.label_encoder = self._encode_labels()
        self.return_images = return_images
        self.root_dir = root_dir

    def __len__(self):
        return len(self.metadata_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.return_images:
            image = self.metadata_frame.iloc[idx, 0]
        else:
            image = io.imread(self.metadata_frame.iloc[idx, 0])
            if self.transform:
                image = self.transform(image)

        label = self.metadata_frame.iloc[idx, 1]
        
        return image, label

    def transform_label(self, label):
        return self.label_encoder.inverse_transform(label)

    def _encode_labels(self):
        label_encoder = LabelEncoder()
        self.metadata_frame['label'] = label_encoder.fit_transform(self.metadata_frame['label'])
        return label_encoder

    def _read_metadata(self, root_dir, return_images=False, include_classes=None, exclude_classes=None):
        files_metadatas = []

        for p in glob.glob(os.path.join(root_dir, '*.jpg')):
            label = p.split('/')[-1].split('_')[0]
            if include_classes and label not in include_classes:
                continue
            if exclude_classes and label in exclude_classes:
                continue
            path = p
            if (return_images):
                im = io.imread(path)
                files_metadatas.append((self.transform(im), label))
            else:
                files_metadatas.append((path, label))

        return pd.DataFrame(files_metadatas, columns=['file', 'label'])


class IrmaDataset(Dataset):
    irma_classmap = LabelEncoder().fit(['BI-RADS I', 'BI-RADS II', 'BI-RADS III', 'BI-RADS IV'])

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