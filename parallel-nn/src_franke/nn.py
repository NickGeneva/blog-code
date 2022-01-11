'''
How to Train Neural Networks in Parallel (From Scratch)
===
Author: Nicholas Geneva (MIT Liscense)
url: https://nicholasgeneva.com/blog/
github: https://github.com/NickGeneva/blog-code
===
'''
import torch.nn as nn

class FCNetwork(nn.Module):

    def __init__(self, in_dim: int, out_dim: int):

        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, out_dim)
        ) 

    def forward(self, x):
        out = self.layers(x)
        return out

    def num_parameters(self):
        count = 0
        for name, param in self.named_parameters():
            # print(name, param.numel())
            count += param.numel()
        return count
