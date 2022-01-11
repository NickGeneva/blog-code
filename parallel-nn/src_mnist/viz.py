'''
How to Train Neural Networks in Parallel (From Scratch)
===
Author: Nicholas Geneva (MIT Liscense)
url: https://nicholasgeneva.com/blog/
github: https://github.com/NickGeneva/blog-code
===
'''
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from torch import nn

Tensor = torch.Tensor

@torch.no_grad()
def plot_prediction(
    y_pred: Tensor, 
    y_target: Tensor, 
    epoch:int = 0, 
    plot_dir:Path = Path('.')
) -> None:
    assert y_pred.size() == y_target.size()
    
    n_plots = y_pred.size(0)
    error = torch.abs(y_target - y_pred)

    plt.close('all')
    fig, ax = plt.subplots(3, n_plots, figsize=(2*n_plots, 6))

    for i in range(n_plots):
        ax[0, i].imshow(y_target[i].cpu().numpy(), cmap=mpl.cm.Greys)
        ax[1, i].imshow(y_pred[i].cpu().numpy(), cmap=mpl.cm.Greys)
        ax[2, i].imshow(error[i].cpu().numpy(), cmap=mpl.cm.viridis)

    ax[0,0].set_ylabel('Target', fontsize=20)
    ax[1,0].set_ylabel('Prediction', fontsize=20)
    ax[2,0].set_ylabel('L1', fontsize=20)

    plt.tight_layout()
    # Save and show figure
    file_name = 'mnist_prediction{:d}.png'.format(epoch)
    plt.savefig(plot_dir / file_name)

@torch.no_grad()
def plot_interpolation(
    model: nn.Module,
    y_target1: Tensor, 
    y_target2: Tensor, 
    epoch:int = 0, 
    plot_dir:Path = Path('.')
) -> None:
    assert y_target1.size() == y_target2.size()
    
    n_steps = 10
    mu1, _ = model.encode(y_target1)
    mu2, _ = model.encode(y_target2)
    mu1 = mu1.squeeze()
    mu2 = mu2.squeeze()

    plt.close('all')
    fig, ax = plt.subplots(1, n_steps, figsize=(2*n_steps,2))

    for i in range(n_steps):
        mu_interp = torch.lerp(mu1, mu2, i*(1./n_steps)).to(mu1.device)
        y_sample = model.decode(mu_interp.unsqueeze(0))
        ax[i].imshow(y_sample.squeeze().cpu().numpy(), cmap=mpl.cm.Greys)

    plt.tight_layout()
    # Save and show figure
    file_name = 'mnist_interp{:d}.png'.format(epoch)
    plt.savefig(plot_dir / file_name)