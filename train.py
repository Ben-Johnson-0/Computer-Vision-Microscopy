import torch
from torch.utils.data import DataLoader
from unet_model import UNet
from CARESDataset import CARESDataset
from utils import collate

# Hyperparameters
NUM_EPOCHS = 10
ALPHA = 1e-3
BATCH_SIZE = 5
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# DEVICE = torch.device('cpu')

fp = "./data/Synthetic_tubulin_gfp/train_data/data_label.npz"


def train_one_epoch(model, dataloader):
    avg_loss = 0.0
    model.train()
    for i, (images, targets) in enumerate(dataloader, 1):
        # Empty the gradients first
        optim.zero_grad()

        # Move to correct device
        images = torch.stack(images, dim=0)
        images = images.to(DEVICE)
        targets = torch.stack(targets, dim=0)
        targets = targets.to(DEVICE)

        # Perform the forward pass
        predict = model(images)
        predict = torch.sigmoid(predict)

        # Loss
        loss = torch.nn.functional.binary_cross_entropy(predict, targets, reduction = 'mean')
        avg_loss += loss.item()

        # Perform the backward pass
        loss.backward()

        # Perform SGD update
        optim.step()

    avg_loss /= len(dataloader)
    return avg_loss


if __name__ == '__main__':

    # Remove randomness
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # Dataset and Dataloader
    ds_train = CARESDataset(fp, normalize=True)
    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, 
                          num_workers=0, collate_fn=collate)

    # model
    model = UNet(n_channels = 1, n_classes = 1)
    model.to(DEVICE)

    # Optimizer
    optim = torch.optim.Adam(model.parameters(), lr = ALPHA)

    # Training
    for i in range(1,NUM_EPOCHS+1):
        print(f"Epoch {i} of {NUM_EPOCHS}")
        avg_loss = train_one_epoch(model, dl_train)
        print(f"  Average loss: {avg_loss}")
    
    print("Complete.")