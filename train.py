import os

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from model.imgvae import ImgVAE
from utils.distributions import gaussian_mixture_batch, link_batch
from utils.dataset import ImageDataset
from utils.functions import vae_loss, minmax_normalize, coords_to_image
from utils.options import get_options
from utils.distributions import link_batch

def plot_losses(losses, opts):
    # Plot data
    epoch_range = np.arange(opts.num_epochs)
    plt.plot(epoch_range, losses['total'], label='Total loss')
    plt.plot(epoch_range, losses['recon'], label='Reconstruction loss')
    plt.plot(epoch_range, losses['kl'], label='KL loss')

    # Label plot
    plt.title(f"Training Loss (epoch_size={opts.epoch_size}, batch_size={opts.batch_size}, l={opts.latent_dim})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Save plot
    result_dir = os.path.join(opts.result_dir, 'plots')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    plt.savefig(os.path.join(result_dir, f'losses_{opts.run_name}.png'), format='png')
    plt.close()

# Utility function for sampling from the VAE
def plot_samples(model, opts, num_samples=5):
    # Sample from model
    # model.eval()
    # sequence_length = opts.graph_size
    
    # with torch.no_grad():
    #     sampled_images = model.sample(num_samples=5, device=opts.device)  # (5,1,100,100)
    # sampled_images = sampled_images.squeeze(1).cpu().numpy()          # (5,100,100)

    # for i in range(5):
    #     plt.imshow(sampled_images[i], cmap='gray')
    #     plt.title(f"Sampled Sequences (epoch_size={opts.epoch_size}, batch_size={opts.batch_size}, l={opts.latent_dim})")
    #     plt.show()

    # # Save plot
    # result_dir = os.path.join(opts.result_dir, 'plots')
    # if not os.path.exists(result_dir):
    #     os.makedirs(result_dir)
    # plt.savefig(os.path.join(result_dir, f'samples_{opts.run_name}.png'), format='png')
    # plt.close()
    model.eval()
    # sequence_length = opts.graph_size

    with torch.no_grad():
        sampled_sequences = model.sample(num_samples=num_samples, device=opts.device)

    # Plot each sequence
    sampled_sequences_np = sampled_sequences.cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, seq in enumerate(sampled_sequences_np):
        x = seq[:, 0]
        y = seq[:, 1]
        ax.scatter(x, y, marker='o', label=f"Sample {i+1}")

    # Label plot
    ax.set_title(f"Sampled Sequences (epoch_size={opts.epoch_size}, batch_size={opts.batch_size}, l={opts.latent_dim})")
    ax.grid(True)
    ax.legend()

    # Save plot
    result_dir = os.path.join(opts.result_dir, 'plots')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    plt.savefig(os.path.join(result_dir, f'samples_{opts.run_name}.png'), format='png')
    plt.close()
   
def plot_vae_samples(model, opts, num_samples=5, device='cuda'):
    """
    Draws 'num_samples' random latent vectors from N(0,I), 
    decodes them, and plots the resulting 100x100 images.
    """
    model.eval()
    with torch.no_grad():
        # Sample from the latent space: shape (num_samples, latent_dim)
        z = torch.randn(num_samples, model.latent_dim, device=device)

        # Decode z into images: shape (num_samples, 1, 100, 100)
        samples = model.decoder(z)
        
        # Optionally apply sigmoid if your decoder does not already do it
        samples = torch.sigmoid(samples)
        
        print(samples.min().item(), samples.max().item())

    # Move to CPU, remove channel-dim => (num_samples, 100, 100)
    samples_np = samples.squeeze(1).cpu().numpy()

    # Plot each image in a row
    fig, axes = plt.subplots(1, num_samples, figsize=(3 * num_samples, 3))
    if num_samples == 1:
        axes = [axes]  # Make it iterable if there's only one sample

    for i, ax in enumerate(axes):
        ax.imshow(samples_np[i], cmap='gray')
        ax.axis('off')
        ax.set_title(f"Sample {i+1}")

    plt.tight_layout()
    plt.show()
    
    result_dir = os.path.join(opts.result_dir, 'plots')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    plt.savefig(os.path.join(result_dir, f'vae_samples_{opts.run_name}.png'), format='png')
    plt.close()
    
# Main run function
def run(opts):    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ImgVAE(opts.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    losses = {
        'total': [],
        'recon': [],
        'kl': []
    }
    
    # training loop
    for epoch in range(opts.num_epochs):
        # Train with random data
        # LINK_VALUES = [1, 5, 10, 15]
        # lbatch_size = opts.epoch_size // len(LINK_VALUES)
        # data = np.zeros((opts.epoch_size, opts.graph_size, opts.element_dim))
        # for i, link_size in enumerate(LINK_VALUES):
        #     data[i*lbatch_size:(i+1)*lbatch_size] = link_batch(lbatch_size, opts.graph_size, link_size=link_size, noise=0.05)
        data = gaussian_mixture_batch(opts.epoch_size, opts.graph_size, cdist=50)

        idx = np.arange(opts.epoch_size)
        np.random.shuffle(idx)
        data = data[idx]

        # sorting data by increasing values of x
        sorted_indicies = np.argsort(data[:, :, 0], axis=1)
        sorted_data = np.take_along_axis(data, sorted_indicies[:, :, None], axis=1)

        # confirm data was sorted properly
        x_vals = sorted_data[:, :, 0] 
        assert np.all(np.diff(x_vals, axis=1) >= 0), "x values are not sorted in non decreasing order"

        # normalize and convert to img
        images = coords_to_image(minmax_normalize(sorted_data))
        
        dataset = ImageDataset(images)
        dataloader = DataLoader(dataset, batch_size=opts.batch_size, shuffle=True, pin_memory=True)
        for batch in dataloader:
            batch = batch.to(opts.device)
            
            optimizer.zero_grad()
            out, mean, log_var = model(batch)
            loss, rl, kl = vae_loss(out, batch, mean, log_var, epoch)
            loss.backward()
            optimizer.step()
        
        # Save and log losses
        losses['total'].append(loss.item())
        losses['recon'].append(rl.item())
        losses['kl'].append(kl.item())
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Recon: {rl.item():.4f}, KL: {kl.item():.4f}")

    # Plotting results
    plot_losses(losses, opts)
    plot_samples(model, opts)
    plot_vae_samples(model, opts, num_samples=5, device=device)
    print("Training complete and plots saved")

    # Save model
    result_dir = os.path.join(opts.result_dir, 'models')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    model_path = os.path.join(result_dir, f'model_{opts.run_name}.pth')
    model = model.cpu()
    torch.save(model.state_dict(), model_path)
    print("Model saved")

# Program entrypoint
if __name__ == "__main__":
    run(get_options())
