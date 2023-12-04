import torch
import numpy as np
from torch.utils.data import Dataset


class CARESDataset(Dataset):
    ''' Cell dataset for CARES data '''

    def __init__(self, npz_file, normalize=True):
        self.normalize = normalize

        with np.load(npz_file, mmap_mode='r') as data:
            self.length = len(data['Y'])
        
        self.data = np.load(npz_file, mmap_mode='r')


    def __getitem__(self, idx):
        ''' Get the image and the target'''
        image = torch.tensor(self.data['X'][idx])
        target = torch.tensor(self.data['Y'][idx])

        if self.normalize:
            target = torch.sigmoid(target)

        return image, target

    def __len__(self):
        return self.length


if __name__ == '__main__':

    fp = "./data/Projection_Flywing/train_data/data_label.npz"
    ds_train = CARESDataset(fp, normalize=False)
    print(len(ds_train))

    image, target = ds_train[0]
    print(image.shape)
    print(target.shape)
