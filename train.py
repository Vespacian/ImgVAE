import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms
import seaborn as sns
import torch.nn as nn


from imgvae import VAE_Conv
from functions import validate_one_epoch, train_one_epoch
# from utils.dataset import CoordinateDataset
# from utils.functions import vae_loss
# from utils.options import get_options
# from utils.distributions import link_batch


# Utility function for plotting
def plot_losses(losses, opts):
    # Plot data
    epoch_range = np.arange(opts.num_epochs)
    plt.plot(epoch_range, losses['total'], label='Total loss')
    plt.plot(epoch_range, losses['recon'], label='Reconstruction loss')
    plt.plot(epoch_range, losses['kl'], label='KL loss')

    # Label plot
    plt.title(f"Training Loss (h={opts.hidden_dim}, l={opts.latent_dim})")
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
    model.eval()
    sequence_length = opts.graph_size

    with torch.no_grad():
        sampled_sequences = model.sample(num_samples=num_samples, seq_length=sequence_length, device=opts.device)

    # Plot each sequence
    sampled_sequences_np = sampled_sequences.cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, seq in enumerate(sampled_sequences_np):
        x = seq[:, 0]
        y = seq[:, 1]
        ax.scatter(x, y, marker='o', label=f"Sample {i+1}")

    # Label plot
    ax.set_title(f"Sampled Sequences (h={opts.hidden_dim}, l={opts.latent_dim})")
    ax.grid(True)
    ax.legend()

    # Save plot
    result_dir = os.path.join(opts.result_dir, 'plots')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    plt.savefig(os.path.join(result_dir, f'samples_{opts.run_name}.png'), format='png')
    plt.close()
    
def run_sample(batch_size, device, model):
    # define transformations
    transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )   

    transform_test = transforms.Compose(
        [
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    
    # load the cifar10 dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root = './data', train = True,
        # root = path_to_data + "data", train = True,
        download = True, transform = transform_train
        )

    test_dataset = torchvision.datasets.CIFAR10(
        root = './data', train = False,
        # root = path_to_data + "data", train = True,
        download = True, transform = transform_test
        )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size = batch_size,
        shuffle = True
        )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size = batch_size,
        shuffle = False
        )
    
    print(f"Sizes of train_dataset: {len(train_dataset)} and test_dataet: {len(test_dataset)}")
    print(f"Sizes of train_loader: {len(train_loader)} and test_loader: {len(test_loader)}")
    print(f"len(train_loader) = {len(train_loader)} & len(test_loader) = {len(test_loader)}")
    print(len(train_dataset) / batch_size, len(test_dataset) / batch_size)
    
    # Get some random batch of training images & labels-
    images, labels = next(iter(train_loader))

    # You get 64 images due to the specified batch size-
    print(f"images.shape: {images.shape} & labels.shape: {labels.shape}")
    
    images = images / 2 + 0.5
    images = np.transpose(images.numpy(), (0, 2, 3, 1))
    print(images.shape)
    
    # plot a sample image
    print("sample image: ", images[0].shape)
    plt.figure(figsize = (7, 6))
    plt.imshow(images[0])
    plt.show()
    
    # Visualize 10 images from training set-
    for i in range(10):
        plt.subplot(2, 5, i + 1)    # 2 rows & 5 columns
        plt.imshow(images[i])
        
    plt.suptitle("Sample CIFAR-10 training images")
    plt.show()
    
    # prechecks
    print(model)
    
    # Count number of layer-wise parameters and total parameters-
    tot_params = 0
    for param in model.parameters():
        print(f"layer.shape = {param.shape} has {param.nelement()} parameters")
        tot_params += param.nelement()
    
    print(f"Total number of parameters in VAE Dense model = {tot_params}")
    
    # sanity checks
    for x in model.hidden2mu.parameters():
        print(x.shape, x.nelement())
        
    log_var_wts = model.hidden2log_var.weight
    mu_wts = model.hidden2mu.weight
    
    mu_wts = mu_wts.detach().cpu().numpy()
    log_var_wts = log_var_wts.detach().cpu().numpy()
    
    mu_wts.shape, log_var_wts.shape
    sns.displot(data = mu_wts.flatten(), bins = int(np.ceil(np.sqrt(mu_wts.size))))
    plt.title("mu randomly initialized - Visualization")
    plt.show()
    
    sns.displot(data = log_var_wts.flatten(), bins = int(np.ceil(np.sqrt(log_var_wts.size))))
    plt.title("log variance randomly initialized - Visualization")
    plt.show()
    
    # images = images.to(device)
    # print(images.shape)
    # print(images.min(), images.max())
    # recon_images, mu, log_var = model(images)
    # print(recon_images.shape, mu.shape, log_var.shape)
    # print(recon_images.min().detach().cpu().numpy(), recon_images.max().detach().cpu().numpy())
    

    
    
    
    

# Main run function
def run():    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Available device is {device}')
    
    # hyperparams
    num_epochs = 50
    batch_size = 32
    learning_rate = 0.001
    
    # model stuff now
    model = VAE_Conv(latent_space = 200).to(device)
    
    # run the sample from the example
    # run_sample(batch_size, device, model)
    
    
    transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )   

    transform_test = transforms.Compose(
        [
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    # load the cifar10 dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root = './data', train = True,
        # root = path_to_data + "data", train = True,
        download = True, transform = transform_train
        )

    test_dataset = torchvision.datasets.CIFAR10(
        root = './data', train = False,
        # root = path_to_data + "data", train = True,
        download = True, transform = transform_test
        )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size = batch_size,
        shuffle = True
        )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size = batch_size,
        shuffle = False
        )
    # Specify alpha - Hyperparameter to control the importance of reconstruction
    # loss vs KL-Divergence Loss-
    alpha = 1
    # Python dict to contain training metrics-
    train_history = {}
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    criterion = nn.MSELoss(reduction = 'sum')
    
    # training loop
    for epoch in range(1, num_epochs + 1):
        # Train model for 1 epoch-
        train_epoch_loss, mu_train, logvar_train = train_one_epoch(
            model = model, dataloader = train_loader,
            alpha = alpha, device=device, 
            train_dataset = train_dataset, optimizer = optimizer, criterion=criterion
        )
        
        # Get validation metrics-
        val_epoch_loss, mu_val, logvar_val = validate_one_epoch(
            model = model, dataloader = test_loader,
            alpha = alpha, device=device, test_dataset=test_dataset, criterion=criterion
        )
        
        # Retrieve model performance metrics-
        logvar_train = logvar_train.mean().detach().cpu().numpy()
        logvar_val = logvar_val.mean().detach().cpu().numpy()
        mu_train = mu_train.mean().detach().cpu().numpy()
        mu_val = mu_val.mean().detach().cpu().numpy()

        # Store model performance metrics in Python3 dict-
        train_history[epoch] = {
            'train_loss': train_epoch_loss,
            'val_loss': val_epoch_loss,
            'train_logvar': logvar_train,
            'val_logvar': logvar_val,
            'train_mu': mu_train,
            'val_mu': mu_val
        }

        print(f"Epoch = {epoch}; train loss = {train_epoch_loss:.4f},"
        f"test loss = {val_epoch_loss:.4f}, train_logvar = {logvar_train:.6f}"
        f", train_mu = {mu_train:.6f}, val_logvar = {logvar_val:.6f} &"
        f" val_mu = {mu_val:.6f}")


# Program entrypoint
if __name__ == "__main__":
    run()
