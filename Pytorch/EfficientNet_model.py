import torch
import torch.nn as nn
from math import ceil


base_model = [
    # expand_ratio, channels, repeats, stride, kernel_size
    [1,16,1,1,3],
    [6,24,2,2,3],
    [6,40,2,2,5],
    [6,80,3,2,3],
    [6,1112,3,1,5],
    [6,192,4,2,5],
    [6,320,1,1,3],


]


phi_values = {
    # tuple of : (phi_value, resuolution, drop_rate)
    "b0": (0, 244,0.2), # alpha, beta, gamma, depth = alpha ** phi
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2,300,0.3),
    "b4": (3,380,0.4),
    "b5": (2,456,0.4),
    "b6": (2,528,0.5),
    "b7": (2,600,0.6),
}

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super(CNNBlock,self).__init__()
        self.cnn = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups = groups, # depth wise convolution
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU() # Silu  == Swish

    def forward(self, x):
        return  self.silu(self.bn(self.cnn(x)))



class SqueezeExcitation(nn.Module): # to compute the attention score for each of the channels
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d
        )


class InvertedResidualBlock(nn.Module):
    pass

class EfficientNet(nn.Module):
    pass