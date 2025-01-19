import torch
from torch.utils.data import Dataset

class CoordinateDataset(Dataset):
    def __init__(self, data_array):
        self.data = data_array
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)

class ImageDataset(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        # Convert to torch float
        img = torch.from_numpy(self.images[idx])  # shape (1, 100, 100)
        return img