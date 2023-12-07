import torch
from torch.utils.data import DataLoader

from utils import save_img, tensor_to_patches, patches_to_tensor
from CAREDataset import CAREDataset
from denoise_training import test

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
SAVE_IMAGES = True
BATCH_SIZE = 8

model_path = "tubulin_model_100epoch.pt"    # Trained on Synthetic tubulin gfp

model = torch.load(model_path).to(DEVICE)


gfp_dataset = CAREDataset("./data/Synthetic_tubulin_gfp/train_data/data_label.npz", normalize=True)
gfp_dataloader = DataLoader(gfp_dataset, batch_size=BATCH_SIZE, shuffle=True)
gfp_loss = test(model, gfp_dataloader, -1)
print(f"GFP Data loss: {gfp_loss}")

granule_dataset = CAREDataset("./data/Synthetic_tubulin_granules/train_data/channel_granules/data_label.npz", normalize=True)
granule_dataloader = DataLoader(granule_dataset, batch_size=BATCH_SIZE, shuffle=True)
granule_loss = test(model, granule_dataloader, -2)
print(f"Granule Data loss: {granule_loss}")


tubulin_dataset = CAREDataset("./data/Synthetic_tubulin_granules/train_data/channel_tubules/data_label.npz", normalize=True)
tubulin_dataloader = DataLoader(tubulin_dataset, batch_size=BATCH_SIZE, shuffle=True)
tubulin_loss = test(model, tubulin_dataloader, -3)
print(f"Tubulin Data loss: {tubulin_loss}")

