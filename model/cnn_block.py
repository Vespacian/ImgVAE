import torch.nn as nn
import torch.nn.functional as F

class cnn_block(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(cnn_block, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels = input_channels, out_channels = output_channels,
            kernel_size = 3, stride = 1,
            padding = 1, bias = True
        )
        self.conv2 = nn.Conv2d(
            in_channels = output_channels, out_channels = output_channels,
            kernel_size = 3, stride = 1,
            padding = 1, bias = True
        )
        self.pool = nn.MaxPool2d(
            kernel_size = 2, stride = 2
        )
  
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        return self.pool(x)