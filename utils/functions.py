import torch
import math
import numpy as np

def kl_scheduler(epoch, max=-1):
    val = 0.5 * (math.tanh((epoch/200) - 2.5) + 1)
    return min(max, val) if max >= 0 else val

# Loss function for VAE
# Returns tuple of (total_loss, recon_loss, kl_loss)
def vae_loss(recon, x, mean, log_var, epoch):
    x_noise = x + 0.01 * torch.randn_like(x)
    recon_loss = ((recon - x_noise)**2).mean()
    kl_div = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
    return recon_loss + kl_scheduler(epoch) * kl_div, recon_loss, kl_div

# converting each batch of TSP points into an (img_size x img_size) image
def coords_to_image(coord_batch, image_size=100):
    """
    coord_batch: (B, 50, 2) array of coordinates in [0,1].
    Returns: (B, 1, image_size, image_size) array of float32 images.
    """
    B, G, _ = coord_batch.shape  # G=50
    images = np.zeros((B, 1, image_size, image_size), dtype=np.float32)
    
    for i in range(B):
        for j in range(G):
            x, y = coord_batch[i, j]  # each in [0,1]
            px = int(round(x * (image_size - 1)))
            py = int(round(y * (image_size - 1)))
            # Clip to ensure in-bounds
            px = np.clip(px, 0, image_size - 1)
            py = np.clip(py, 0, image_size - 1)
            images[i, 0, py, px] = 1.0
    
    return images

# normalize the TSP points before coords_to_image conversion
def minmax_normalize(coords, eps=1e-6):
    """
    coords: (B, G, 2) array of coordinates, 
            e.g. B=epoch_size or batch_size, G=50 points per instance

    Returns: (B, G, 2) array of normalized coordinates in [0,1]^2
    """
    B, G, _ = coords.shape
    normalized = np.zeros_like(coords, dtype=np.float32)

    for i in range(B):
        x_vals = coords[i, :, 0]
        y_vals = coords[i, :, 1]

        x_min, x_max = x_vals.min(), x_vals.max()
        y_min, y_max = y_vals.min(), y_vals.max()

        # Avoid zero division
        width  = max(x_max - x_min, eps)
        height = max(y_max - y_min, eps)

        # Scale each coordinate to [0,1]
        normalized[i, :, 0] = (x_vals - x_min) / width
        normalized[i, :, 1] = (y_vals - y_min) / height

    return normalized