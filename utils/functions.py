import torch
from tqdm import tqdm
import math

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
