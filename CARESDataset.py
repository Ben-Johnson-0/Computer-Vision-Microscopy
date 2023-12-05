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
    dataset = CARESDataset(fp, normalize=True)
    dl_train = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True)

    for (image, target) in dl_train:
        print(image.shape)
        print(target.shape)

        # for i in range(image.shape[2]):
        #     save_image(image[0,0,i,:,:], f'image{i}.png')

        max_proj = torch.max(image[0,0,:,:,:], dim=0)
        print(max_proj.values)
        save_image(max_proj.values, 'max-projection.png')

        image_fft = torch.fft.fft2(image, dim=(-2, -1))
        # Example: Keep only the low-frequency components (e.g., 10% of the highest magnitudes)
        magnitude = torch.abs(image_fft)
        threshold = torch.topk(magnitude.view(-1), int(0.1 * magnitude.numel()), largest=True).values.min()
        image_fft[magnitude < threshold] = 0

        # Step 3: Apply inverse FFT to obtain the denoised image
        denoised_image = torch.fft.ifft2(image_fft, dim=(-2, -1)).real

        for i in range(denoised_image.shape[2]):
            save_image(denoised_image[0,0,i,:,:], f'image{i}.png')

        save_image(torch.max(denoised_image[0,0,:,:,:],dim=0 ).values, 'fft-max-projection.png')

        save_image(target[0,0,:,:], 'target.png')
        exit()