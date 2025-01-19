import torch
import torch.nn as nn

from model.encoder import Conv6_Encoder
from model.decoder import Conv6_Decoder

# get the dims right, 
# cifar10 - color images, assume 100x100 img for now
# need convolutional code
# deconv - the inverse of convolution in the decoder (you should see this in the decoder)
# in pytorch convo2D - something like that

class ImgVAE(nn.Module):
    def __init__(self, latent_dim = 16):
        super(ImgVAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        self.encoder = Conv6_Encoder(latent_space = self.latent_dim)
        self.hidden2mu = nn.Linear(in_features = self.latent_dim, out_features = latent_dim, bias = True)
        self.hidden2log_var = nn.Linear(in_features = self.latent_dim, out_features = latent_dim, bias = True)
        self.decoder = Conv6_Decoder(latent_space = self.latent_dim)
        
    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + (std * eps)
    
    def forward(self, x):
        new_x = self.encoder(x)
        mean = self.hidden2mu(new_x)
        log_var = self.hidden2log_var(new_x)
        z = self.reparameterize(mean, log_var)
        recon_data = torch.sigmoid(self.decoder(z))
        return recon_data, mean, log_var

    def sample(self, num_samples=1, device='cpu'):
        z = torch.randn(num_samples, self.latent_dim).to(device)
        with torch.no_grad():
            generated = self.decoder(z)
            generated = torch.sigmoid(generated)
        return generated
