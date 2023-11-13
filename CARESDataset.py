import torch
import numpy as np
from torch.utils.data import Dataset


class CARESDataset(Dataset):
    ''' Cell dataset for CARES data '''

    def __init__(self, npz_file, normalize=True):
        self.normalize = normalize

        data = np.load(npz_file)
        self.images = data['X']
        self.targets = data['Y']

    def __getitem__(self, idx):
        ''' Get the image and the target'''
        image = torch.tensor(self.images[idx])
        target = torch.tensor(self.targets[idx])

        if self.normalize:
            # image = torch.sigmoid(image)
            target = torch.sigmoid(target)

        return image, target

    def __len__(self):
        return self.images.shape[0]


if __name__ == '__main__':
    # fp = "./data/Denoising_Tribolium/train_data/data_label.npz"
    fp = "./data/Synthetic_tubulin_gfp/train_data/data_label.npz"
    ds_train = CARESDataset(fp, normalize=False)

    image, target = ds_train[0]
    print(image.shape)
    print(target.shape)
