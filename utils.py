import torch
import numpy as np
import matplotlib.pyplot as plt

# Save an image
def save_img(img, title, file_name, num_rows, num_cols):
    example = np.reshape(img, (num_rows, num_cols))
    plt.clf()
    plt.matshow(example)
    plt.title(title)
    plt.savefig(file_name)

# Apply FFT then iFFT to denoise an image/stack of images
def fft_denoise(image_tensor, ratio=0.1):
    image_fft = torch.fft.fft2(image_tensor, dim=(-2, -1))
    
    magnitude = torch.abs(image_fft)
    threshold = torch.topk(magnitude.view(-1), int(0.1 * magnitude.numel()), largest=True).values.min()
    image_fft[magnitude < threshold] = 0

    # Step 3: Apply inverse FFT to obtain the denoised image
    denoised_image = torch.fft.ifft2(image_fft, dim=(-2, -1)).real
    return denoised_image