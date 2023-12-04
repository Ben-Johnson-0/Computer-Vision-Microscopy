# Quick, simple visualization of the training data for the 2D data

import numpy as np
import matplotlib.pyplot as plt

fp = "../data/Synthetic_tubulin_gfp/train_data/data_label.npz"
data = np.load(fp)

for k in data.keys():
     print(k)

print('X', data['X'].shape)
print('Y', data['Y'].shape)

for i in range(data['X'].shape[0]):
    img1 = np.squeeze(data['X'][i])
    img2 = np.squeeze(data['Y'][i])
    plt.clf()
    plt.subplot(211)
    plt.imshow(img1)
    plt.title(f"Image {i}")
    plt.subplot(212)
    plt.imshow(img2)
    plt.show()
