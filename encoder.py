import torch.nn as nn
from cnn_block import cnn_block
import torch.nn as nn
import torch.nn.functional as F
import torch

class Conv6_Encoder(nn.Module):
    def __init__(self, latent_space):
        super(Conv6_Encoder, self).__init__()
        self.latent_space = latent_space
        
        self.block1 = cnn_block(input_channels = 3, output_channels = 64)
        self.block2 = cnn_block(input_channels = 64, output_channels = 128)
        self.block3 = cnn_block(input_channels = 128, output_channels = 256)
        self.conv_layer = nn.Conv2d(
            in_channels = 256, out_channels = 256,
            kernel_size = 3, stride = 1,
            padding = 0, bias = True
        )
        self.dense_layer1 = nn.Linear(
            in_features = 1024, out_features = 500,
            bias = True
        )
        self.op_layer = nn.Linear(
            in_features = 500, out_features = self.latent_space,
            bias = True
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = F.leaky_relu(self.conv_layer(x))
        x = torch.flatten(x, start_dim = 1)
        x = F.leaky_relu(self.dense_layer1(x))
        x = F.leaky_relu(self.op_layer(x))
        return x