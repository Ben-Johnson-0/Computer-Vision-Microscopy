
# Collate function for the DataLoader
def collate(x):
    return tuple(zip(*x))

