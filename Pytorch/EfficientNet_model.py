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
    "b1":(0.5, )



}