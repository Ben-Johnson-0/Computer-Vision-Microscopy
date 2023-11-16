import numpy as np
import matplotlib.pyplot as plt

# Collate function for the DataLoader
def collate(x):
    return tuple(zip(*x))

# Save an image
def save_img(img, title, file_name, num_rows, num_cols):
    example = np.reshape(img, (num_rows, num_cols))
    plt.clf()
    plt.matshow(example)
    plt.title(title)
    plt.savefig(file_name)