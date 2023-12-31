import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from unet_model import UNet
from CAREDataset import CAREDataset
from utils import save_img

# Hyperparameters
SEED = 1
NUM_EPOCHS = 100
EPOCHS_PER_TEST = 20
ALPHA = 1e-3
BATCH_SIZE = 16
PERCENT_TRAIN = 0.8
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
SAVE_IMAGES = False
MODEL_FILENAME = 'tubulin_model_testing.pt'

fp = "./data/Synthetic_tubulin_gfp/train_data/data_label.npz"
# fp = "./data/Synthetic_tubulin_granules/train_data/channel_tubules/data_label.npz"


def train_one_epoch(model, dataloader):
    """ Train for one epoch, returns the epoch's average loss. """
    avg_loss = 0.0
    model.train()
    for i, (images, targets) in enumerate(dataloader, 1):
        # Empty the gradients first
        optim.zero_grad()

        # Move to correct device
        images = images.to(DEVICE).float()
        targets = targets.to(DEVICE).float()
                
        # Perform the forward pass
        predict = model(images)

        # Loss
        predict = torch.sigmoid(predict)
        loss = nn.functional.binary_cross_entropy(predict, targets, reduction = 'mean')
        avg_loss += loss.item()

        # Perform the backward pass
        loss.backward()

        # Perform SGD update
        optim.step()

    avg_loss /= len(dataloader)
    return avg_loss


def test(model, dataloader, epoch):
    """ Run a test epoch, returns the test run's  average MSE Loss. """
    avg_loss = 0.0
    model.eval()
    for i, (images, targets) in enumerate(dataloader, 1):
        # Move to correct device
        images = images.to(DEVICE).float()
        targets = targets.to(DEVICE).float()

        # Perform the forward pass
        predict = model(images)
        predict = torch.sigmoid(predict)

        # Save first prediction, it's target, and it's original input
        if i == 1 and SAVE_IMAGES:
            save_img(images[0, 0, :, :].detach().cpu().numpy(), f"Epoch {epoch} input", f"outputs/{epoch}_input.png", images.shape[-2], images.shape[-1])
            save_img(targets[0, 0, :, :].detach().cpu().numpy(), f"Epoch {epoch} target", f"outputs/{epoch}_target.png", targets.shape[-2], targets.shape[-1])
            save_img(predict[0, 0, :, :].detach().cpu().numpy(), f"Epoch {epoch} denoised", f"outputs/{epoch}_denoising.png", predict.shape[-2], predict.shape[-1])

        # Loss
        mse_loss = nn.MSELoss()
        loss = mse_loss(predict, targets)
        avg_loss += loss.item()
        
    avg_loss /= len(dataloader)
    return avg_loss


if __name__ == '__main__':

    # Remove randomness
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    # Dataset and Dataloader
    dataset = CAREDataset(fp, normalize=True)
    generator = torch.Generator().manual_seed(SEED)
    ds_train, ds_test = random_split(dataset, [PERCENT_TRAIN, 1-PERCENT_TRAIN], generator=generator)
    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

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
        if i % EPOCHS_PER_TEST == 0:
            acc = test(model, dl_test, i)
            print(f"Test MSE Loss: {acc}")
        
    print(f"Saving model as \"{MODEL_FILENAME}\"")
    torch.save(model, MODEL_FILENAME)
    print("Complete.")