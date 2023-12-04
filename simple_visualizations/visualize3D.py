# Quick, simple visualization of the training data for the 3D data

import numpy as np
import matplotlib.pyplot as plt

fp = "../data/Denoising_Tribolium/train_data/data_label.npz"
data = np.load(fp)

for k in data.keys():
     print(k)

print('X', data['X'].shape)
print('Y', data['Y'].shape)

for i in range(data['X'].shape[0]):
     for k in range(data['X'].shape[2]):
          img1 = np.squeeze(data['X'][i][0][k])
          img2 = np.squeeze(data['Y'][i][0][k])
          plt.clf()
          plt.subplot(211)
          plt.imshow(img1)
          plt.title(f"Image {i}")
          plt.subplot(212)
          plt.imshow(img2)
          plt.show()
