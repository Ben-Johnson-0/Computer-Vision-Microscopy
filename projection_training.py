import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from CAREDataset import CAREDataset
from rnn_model import ProjectionRNN
from utils import save_img

# TODO
#  Solve the all batches give the same value bug

# Hyperparameters
IMAGE_SIZE = 64*64
N_HIDDEN = 2048
BATCH_SIZE = 8
REC_LAYERS = 1
ALPHA = 1e-5
EPOCHS = 1000
PERCENT_TRAIN = 0.8
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Remove randomness
torch.manual_seed(1)
torch.cuda.manual_seed(1)

# Create model
print("Creating model")
model = ProjectionRNN(input_size=IMAGE_SIZE, hidden_size=N_HIDDEN, output_size=IMAGE_SIZE, num_layers=REC_LAYERS)
model.to(DEVICE)

# Optimizer
optim = torch.optim.Adam(model.parameters(), lr = ALPHA)

# Dataset and Dataloader
print("Loading Dataset")
# fp = "./data/Projection_Flywing/train_data/data_label.npz"
fp = "./data/mini_Projection_Flywing"
dataset = CAREDataset(fp, normalize=False)
generator = torch.Generator().manual_seed(1)
ds_train, ds_test = random_split(dataset, [PERCENT_TRAIN, 1-PERCENT_TRAIN], generator=generator)
dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

print("Training model")
avg_loss = 0.0
model.train()
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1} of {EPOCHS}")
    for i, (images, targets) in enumerate(dl_train, 1):
        # Empty the gradients first
        optim.zero_grad()

        # Images shape: [Batch, Channels, Sequence_length, x, y]
        images = images.to(DEVICE).float()
        images = torch.squeeze(images, dim=1)
        # Images reshape: [Batch, Sequence_length, x*y]
        images = images.reshape(images.size(0), images.size(1), -1)

        # Targets shape: [Batch, Channels, x, y]
        targets = targets.to(DEVICE).float()
        targets = torch.squeeze(targets, dim=1)
        # Targets reshape: [Batch, x*y]
        targets = targets.reshape(targets.size(0), -1)

        # Perform the forward pass
        predict = model(images)

        # Loss
        mse_loss = nn.MSELoss()
        loss = mse_loss(predict, targets)
        avg_loss += loss.item()

        # Perform the backward pass
        loss.backward()

        # Perform SGD update
        optim.step()

        if i == 1:  #debugging the all batches are the same bug
            for j in range(predict.shape[0]):
                if j < 2:
                    print(predict[j])

    avg_loss /= len(dl_train)
    print(avg_loss)

print("Testing model")
avg_loss = 0.0
model.eval()
with torch.no_grad():
    for i, (images, targets) in enumerate(dl_test, 1):
        # Images shape: [Batch, Channels, Sequence_length, x, y]
        images = images.to(DEVICE).float()
        images = torch.squeeze(images, dim=1)
        # Images reshape: [Batch, Sequence_length, x*y]
        images = images.reshape(images.size(0), images.size(1), -1)

        # Targets shape: [Batch, Channels, x, y]
        targets = targets.to(DEVICE).float()
        targets = torch.squeeze(targets, dim=1)
        # Targets reshape: [Batch, x, y]
        targets = targets.reshape(targets.size(0), -1)

        # Perform the forward pass
        predict = model(images)

        # Loss
        mse_loss = nn.MSELoss()
        loss = mse_loss(predict, targets)
        avg_loss += loss.item()

        # Save example image
        if i == 1 and BATCH_SIZE >= 2:
            save_img(targets[0, :].detach().cpu().numpy(), "Projection Target 0", "outputs/projection_target0.png", 64, 64)
            save_img(predict[0, :].detach().cpu().numpy(), "Projection Prediction 0", "outputs/projection_prediction0.png", 64, 64)
            save_img(targets[1, :].detach().cpu().numpy(), "Projection Target 1", "outputs/projection_target1.png", 64, 64)
            save_img(predict[1, :].detach().cpu().numpy(), "Projection Prediction 1", "outputs/projection_prediction1.png", 64, 64)


avg_loss /= len(dl_test)
print(avg_loss)

print("Saving model...")
torch.save(model, 'projection_model.pt')
print("Complete.")