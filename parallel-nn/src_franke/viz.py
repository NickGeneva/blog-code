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

def franke(x1, x2):
    f = 0.75*torch.exp(-(9*x1-2)**2/4.0-(9*x2-2)**2/4.0) + 0.75*torch.exp(-(9*x1+1)**2/49.0-(9*x2+1)**2/10) \
            + 0.5*torch.exp(-(9*x1-7)**2/4.0-(9*x2-3)**2/4.0) - 0.2*torch.exp(-(9*x1-4)**2-(9*x2-7)**2)
    return f

@torch.no_grad()
def plot_prediction(model, epoch:int = 0, plot_dir:Path = Path('.'), device='cpu'):

    plt.close('all')
    fig = plt.figure(figsize=(15,5))
    ax = []
    ax.append(plt.subplot2grid((1, 3), (0, 0), projection='3d'))
    ax.append(plt.subplot2grid((1, 3), (0, 1), projection='3d'))
    ax.append(plt.subplot2grid((1, 3), (0, 2), projection='3d'))

    x, y = np.meshgrid(np.linspace(-1, 1, 200), np.linspace(-1, 1, 200))
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    target = franke(x.view(-1), y.view(-1)).reshape(200, 200)

    input = torch.stack([x.view(-1), y.view(-1)], dim=1).to(device)
    pred = model(input).squeeze().cpu().reshape(200, 200)
    error = torch.abs(target - pred)

    # Appearently matplotlib sucks at 3d plotting and zorders are not fixable
    ax[0].plot_surface(x.numpy(), y.numpy(), target.numpy(), cmap=mpl.cm.inferno, alpha=1.0, vmax=1.0, vmin=-0.2)
    ax[1].plot_surface(x.numpy(), y.numpy(), pred.numpy(), cmap=mpl.cm.inferno, alpha=1.0, vmax=1.0, vmin=-0.2)
    ax[2].plot_surface(x.numpy(), y.numpy(), error.numpy(), cmap=mpl.cm.viridis, alpha=1.0)

    for ax0 in ax:
        ax0.view_init(elev=25., azim=60)
        ax0.set_xlabel('x')
        ax0.set_ylabel('y')
        ax0.set_zlabel('z')
        ax0.set_zlim([-0.2,1])

    ax[2].set_zlim([0,1])

    plt.tight_layout()
    ax[0].set_title('Target', y=1.0, pad=-5, fontsize=20)
    ax[1].set_title('Prediction', y=1.0, pad=-5, fontsize=20)
    ax[2].set_title('L1', y=1.0, pad=-5, fontsize=20)
    # Save and show figure
    file_name = 'franke_prediction{:d}.png'.format(epoch)
    plt.savefig(plot_dir / file_name)