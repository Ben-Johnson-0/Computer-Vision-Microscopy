import torch
import numpy as np
from torch.utils.data import Dataset
from zipfile import ZipFile
from os import path

class CARESDataset(Dataset):
    ''' Cell dataset for CARES data '''

    def __init__(self, np_file, normalize=True):
        """ np_file can be a .npz file or the source directory of the X.npy and Y.npy files """
        self.normalize = normalize

        # Due to some weird interactions with mmap and npz files (they don't work together),
        #  we have to unzip the files so that it doesn't try to load the entire file at once while training
        if (len(np_file) > 4 and np_file[-4:] == '.npz'):
            outpath = '/'.join(np_file.split('/')[:-1]) + '/data_labels'
            if (not path.exists(outpath)):
                with ZipFile(np_file, 'r') as f:
                    f.extractall(path=outpath)
            np_file = outpath

        self.images = np.load(np_file+'/X.npy', mmap_mode='r')
        self.targets = np.load(np_file+'/Y.npy', mmap_mode='r')
        self.length = len(self.targets)

    def __getitem__(self, idx):
        ''' Get the image and the target'''
        image = torch.from_numpy(self.images[idx])
        target = torch.from_numpy(self.targets[idx])

        if self.normalize:
            target = torch.sigmoid(target)

        return image, target

    def __len__(self):
        return self.length

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchvision.utils import save_image

    fp = "./data/Projection_Flywing/train_data/data_label.npz"
    dataset = CARESDataset(fp, normalize=False)

    mini = dataset[:1000]
    print(mini[0].shape)
    print(mini[1].shape)
    np.save("mini_Projection_Flywing/X.npy", mini[0])
    np.save("mini_Projection_Flywing/Y.npy", mini[1])