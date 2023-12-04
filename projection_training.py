import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from CARESDataset import CARESDataset
from rnn_model import ProjectionRNN
from utils import collate

# Remove randomness
torch.manual_seed(1)
torch.cuda.manual_seed(1)

IMAGE_SIZE = 64*64
N_HIDDEN = 64
EPOCHS = 5
ALPHA = 1e-3
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Create model
print("Creating model")
model = ProjectionRNN(input_size=IMAGE_SIZE, hidden_size=N_HIDDEN, output_size=IMAGE_SIZE, num_layers=1)
model.to(DEVICE)

# Optimizer
optim = torch.optim.Adam(model.parameters(), lr = ALPHA)

# Dataset and Dataloader
print("Loading Dataset")
fp = "./data/Projection_Flywing/train_data/data_label.npz"
dataset = CARESDataset(fp, normalize=True)
dl_train = DataLoader(dataset, batch_size=5, shuffle=True)

print("Training model")
avg_loss = 0.0
model.train()
for epoch in range(EPOCHS):
    print(f"Epoch {epoch} of {EPOCHS}")
    for i, (images, targets) in enumerate(dl_train, 1):
        # Empty the gradients first
        optim.zero_grad()

        # Images shape: [Batch, Channels, Sequence_length, x, y]
        # Images reshape: [Batch, Sequence_length, x*y*channels]
        images = images.reshape(images.size(0), images.size(2), -1)
        images = images.to(DEVICE).float()

        # Targets shape: [Batch, Channels, x, y]
        # Targets reshape: [Batch, x*y*channels]
        targets = targets.reshape(targets.size(0), -1)
        targets = targets.to(DEVICE).float()

        # Perform the forward pass
        predict = model(images)

        # Loss
        mse_loss = nn.MSELoss()
        loss = mseloss(predict, targets)
        avg_loss += loss.item()

        # Perform the backward pass
        loss.backward()

        # Perform SGD update
        optim.step()

    avg_loss /= len(dl_train)
    print(avg_loss)

print("Saving model...")
torch.save(model, 'projection_model.pt')
print("Complete.")