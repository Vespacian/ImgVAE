import torch.nn as nn
from cnn_block import cnn_block
import torch.nn as nn
import torch.nn.functional as F
import torch

class Conv6_Decoder(nn.Module):
    def __init__(self, latent_space):
        super(Conv6_Decoder, self).__init__()
        self.latent_space = latent_space
    
        self.dense_layer1 = nn.Linear(
            in_features = self.latent_space, out_features = 500,
            bias = True
        )
        self.dense_layer2 = nn.Linear(
            in_features = 500, out_features = 1024,
            bias = True
        )
        self.conv_tran_layer = nn.ConvTranspose2d(
            in_channels = 256, out_channels = 256,
            kernel_size = 4, stride = 2,
            padding = 1
        )
        self.conv_tran_layer1 = nn.ConvTranspose2d(
            in_channels = 256, out_channels = 256,
            kernel_size = 4, stride = 2,
            padding = 1
        )
        self.conv_tran_layer2 = nn.ConvTranspose2d(
            in_channels = 256, out_channels = 256,
            kernel_size = 3, stride = 1,
            padding = 1
        )
        self.conv_tran_layer3 = nn.ConvTranspose2d(
            in_channels = 256, out_channels = 128,
            kernel_size = 4, stride = 2,
            padding = 1
        )
        self.conv_tran_layer4 = nn.ConvTranspose2d(
            in_channels = 128, out_channels = 128,
            kernel_size = 3, stride = 1,
            padding = 1
        )
        self.conv_tran_layer5 = nn.ConvTranspose2d(
            in_channels = 128, out_channels = 64,
            kernel_size = 4, stride = 2,
            padding = 1
        )
        self.conv_tran_layer6 = nn.ConvTranspose2d(
            in_channels = 64, out_channels = 64,
            kernel_size = 3, stride = 1,
            padding = 1
        )
        self.output_conv_layer = nn.ConvTranspose2d(
            in_channels = 64, out_channels = 3,
            kernel_size = 3, stride = 1,
            padding = 1
        )
        
    
    def forward(self, x):
        x = F.leaky_relu(self.dense_layer1(x))
        x = F.leaky_relu(self.dense_layer2(x))
        # x = x.view(-1, 256, 4, 4)
        x = x.view(-1, 256, 2, 2)
        x = F.leaky_relu(self.conv_tran_layer(x))
        x = F.leaky_relu(self.conv_tran_layer1(x))
        x = F.leaky_relu(self.conv_tran_layer2(x))
        x = F.leaky_relu(self.conv_tran_layer3(x))
        x = F.leaky_relu(self.conv_tran_layer4(x))
        x = F.leaky_relu(self.conv_tran_layer5(x))
        x = F.leaky_relu(self.conv_tran_layer6(x))
        x = self.output_conv_layer(x)
        return x
 