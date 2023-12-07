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

def fft_denoise(image_tensor, ratio=0.1):
    """ Apply FFT then iFFT to denoise an image/stack of images """
    image_fft = torch.fft.fft2(image_tensor, dim=(-2, -1))
    
    magnitude = torch.abs(image_fft)
    threshold = torch.topk(magnitude.view(-1), int(0.1 * magnitude.numel()), largest=True).values.min()
    image_fft[magnitude < threshold] = 0

    # Step 3: Apply inverse FFT to obtain the denoised image
    denoised_image = torch.fft.ifft2(image_fft, dim=(-2, -1)).real
    return denoised_image


def tensor_to_patches(input_tensor, patch_size):
    """ Generate Patches from a 4D tensor """
    batch_size, channels, height, width = input_tensor.size()
    patches = input_tensor.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # patches shape: [1, 1, 4, 8, 64, 64]
    patches = patches.contiguous().view(batch_size, channels, -1, patch_size, patch_size)
    # patches shape: [1, 1, 32, 64, 64]
    patches = patches.squeeze(dim=0)
    # patches shape: [1, 32, 64, 64]
    patches = patches.permute(1, 0, 2, 3).contiguous().view(-1, channels, patch_size, patch_size)
    # patches shape: [32, 1, 64, 64]

    return patches

def patches_to_tensor(patches, original_size, patch_size):
    """ Return a patched tensor to its original 4D shape """
    # original size: [1,1,256,512]
    batch_size, channels, height, width = original_size

    patches = patches.view(batch_size, channels, -1, patch_size, patch_size)
    # patches shape: [32, 1, 64, 64]
    patches = patches.permute(0, 2, 1, 3, 4).contiguous().view(batch_size, channels, -1, patch_size, patch_size)
    # patches shape: [1, 1, 32, 64, 64]

    unfolded_height = height // patch_size  # 4
    unfolded_width = width // patch_size    # 8
    
    reconstructed = patches.view(batch_size, channels, unfolded_height, unfolded_width, patch_size, patch_size)
    # rec shape: [1, 1, 4, 8, 64, 64] 
    reconstructed = reconstructed.permute(0, 1, 2, 4, 3, 5).contiguous()
    # rec shape: [1, 1, 4, 64, 8, 64]
    reconstructed = reconstructed.view(batch_size, channels, unfolded_height * patch_size, unfolded_width * patch_size)
    # rec shape: [1, 1, 4*64, 8*64] == [1, 1, 256, 512]

    return reconstructed