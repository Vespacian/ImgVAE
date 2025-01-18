import time
import argparse
import torch


def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="img-to-img variational autoencoder, used specifically for generating point configurations")

    # Define and parse arguments
    parser.add_argument('--latent_dim', type=int, default=512, help="Size of the latent dimension of the VAE")
    parser.add_argument('--hidden_dim', type=int, default=1024, help="Size of the hidden dimension of the encoder/decoder")
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs to train on')
    parser.add_argument('--epoch_size', type=int, default=65536, help='Number of instances per epoch during training')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of instances per batch during training')
    parser.add_argument('--graph_size', type=int, default=50, help='Number of points for each graph')
    parser.add_argument('--run_name', type=str, default=None, help='Name to identify the run')
    parser.add_argument('--run_timestamp', action='store_true', help='Add a timestamp to the run name')

    opts = parser.parse_args(args)

    # Modify run name
    if opts.run_name is None:
        opts.run_name = f"run-h{opts.hidden_dim}-l{opts.latent_dim}-b{opts.batch_size}-e{opts.num_epochs}"
    if opts.run_timestamp:
        opts.run_name = f"{opts.run_name}_{time.strftime('%Y%m%dT%H%M%S')}"

    # Custom arguments
    opts.result_dir = 'results'
    opts.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opts.element_dim = 2 # x, y coordinates for every point

    return opts
