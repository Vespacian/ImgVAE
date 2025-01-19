import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F

class Conv6_Decoder(nn.Module):
    def __init__(self, latent_space):
        super(Conv6_Decoder, self).__init__()
        self.latent_space = latent_space
    
        self.dense_layer1 = nn.Linear(
            in_features = self.latent_space, out_features = 500,
            bias = True
        )
        self.dense_layer2 = nn.Linear(
            in_features = 500, out_features = 256*25*25,
            bias = True
        )
        
        # 1) Upsample 25->50
        self.deconv_block1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU()
        )

        # 2) Upsample 50->100
        self.deconv_block2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU()
        )

        # Final => out_channels=1 => (B,1,100,100)
        self.output_conv_layer = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):      
        x = F.leaky_relu(self.dense_layer1(x))
        x = F.leaky_relu(self.dense_layer2(x))
        x = x.view(-1, 256, 25, 25)  # match shape from the encoder

        x = self.deconv_block1(x)   # => (B,128,50,50)
        x = self.deconv_block2(x)   # => (B,64,100,100)
        x = self.output_conv_layer(x)
        return x