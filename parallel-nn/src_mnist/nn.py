'''
How to Train Neural Networks in Parallel (From Scratch)
===
Author: Nicholas Geneva (MIT Liscense)
url: https://nicholasgeneva.com/blog/
github: https://github.com/NickGeneva/blog-code
===
'''
import torch
import torch.nn as nn

class VAENetwork(nn.Module):

    def __init__(self, x_dim: int, h_dim1: int, h_dim2: int, z_dim: int):
        super(VAENetwork, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(x_dim, h_dim1),
            nn.ReLU(),
            nn.Linear(h_dim1, h_dim2),
            nn.ReLU()
        ) 
        self.enc_mu = nn.Linear(h_dim2, z_dim)
        self.enc_std = nn.Linear(h_dim2, z_dim)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim2),
            nn.ReLU(),
            nn.Linear(h_dim2, h_dim1),
            nn.ReLU(),
            nn.Linear(h_dim1, x_dim),
            nn.Sigmoid()
        ) 

    def encode(self, x):
        out = self.encoder(x.view(-1, 784))
        mu = self.enc_mu(out)
        std = self.enc_std(out)
        return mu, std

    def decode(self, z):
        return self.decoder(z).view(-1, 28, 28)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return std*eps + mu 

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.sampling(mu, log_var)
        return self.decode(z), mu, log_var

    def num_parameters(self):
        count = 0
        for name, param in self.named_parameters():
            # print(name, param.numel())
            count += param.numel()
        return count
