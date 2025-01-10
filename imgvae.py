import torch
import torch.nn as nn

# from .encoder import Encoder
# from .decoder import Decoder

class ImgVAE(nn.Module):
    pass

# class VAE(nn.Module):
#     def __init__(self):
#         super(VAE, self).__init__()

#         self.fc1 = nn.Linear(196, 128) #Encoder
#         self.fc21 = nn.Linear(128, 8) #mu
#         self.fc22 = nn.Linear(128, 8) #sigma

#         self.fc3 = nn.Linear(8, 128) #Decoder
#         self.fc4 = nn.Linear(128, 196)
        
#     def encoder(self, x):
#         h = torch.tanh(self.fc1(x))
#         return self.fc21(h), self.fc22(h) # mu, std
    
#     def sampling(self, mu, std): # Reparameterization trick
#         eps1 = torch.randn_like(std)
#         eps2 = torch.randn_like(std)
#         return 0.5*((eps1 * std + mu) + (eps2 * std + mu)) # Using two samples to compute expectation over z

#     def decoder(self, z):
#         h = torch.tanh(self.fc3(z))
#         return torch.sigmoid(self.fc4(h)) 
    
#     def forward(self, x):
#         mu, std = self.encoder(x.view(-1, 196))
#         z = self.sampling(mu, std)
#         return self.decoder(z), mu, std
# model = VAE()
# if torch.cuda.is_available():
#     model.cuda()