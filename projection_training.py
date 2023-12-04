from CARESDataset import CARESDataset
from utils import collate

# Remove randomness
torch.manual_seed(1)
torch.cuda.manual_seed(1)

FRAMES = 50
IMAGE_SIZE = 64*64
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
ALPHA = 1e-3

# Create model
print("Creating model")
n_hidden = 64
model = ProjectionRNN(FRAMES*IMAGE_SIZE, n_hidden, IMAGE_SIZE)
model.to(DEVICE)

# Optimizer
optim = torch.optim.Adam(model.parameters(), lr = ALPHA)

# Dataset and Dataloader
print("Loading Dataset")
fp = "./data/Projection_Flywing/train_data/data_label.npz"
ds_train = CARESDataset(fp, normalize=True)
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=5, shuffle=True, 
                        num_workers=0, collate_fn=collate)

print("Training model")
avg_loss = 0.0
model.train()
for i, (images, targets) in enumerate(dl_train, 1):
    # Empty the gradients first
    optim.zero_grad()

    # Move to correct device
    images = torch.stack(images, dim=0)
    images = images.to(DEVICE).float()
    targets = torch.stack(targets, dim=0)
    targets = targets.to(DEVICE).float()

    # Perform the forward pass
    predict = model(images)

    # Loss
    loss = nn.functional.binary_cross_entropy(predict, targets, reduction = 'mean')
    avg_loss += loss.item()

    # Perform the backward pass
    loss.backward()

    # Perform SGD update
    optim.step()

avg_loss /= len(dl_train)
print(avg_loss)

x = dl_train[0]
out = model(x)
print(out.shape)
print(out)