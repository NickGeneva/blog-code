'''
Into to deep learning Koopman operators
===
Author: Nicholas Geneva (MIT Liscense)
url: https://nicholasgeneva.com/blog/
github: https://github.com/AbsoluteStratos/blog-code
===
'''
import torch
import numpy as np
import os

import matplotlib as mpl
# mpl.use('agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerBase

class HandlerColormap(HandlerBase):
    def __init__(self, cmap, num_stripes=8, **kw):
        HandlerBase.__init__(self, **kw)
        self.cmap = cmap
        self.num_stripes = num_stripes
    def create_artists(self, legend, orig_handle, 
                       xdescent, ydescent, width, height, fontsize, trans):
        stripes = []
        for i in range(self.num_stripes):
            s = Rectangle([xdescent + i * width / self.num_stripes, ydescent], 
                          width / self.num_stripes, 
                          height, 
                          fc=self.cmap((2 * i + 1) / (2 * self.num_stripes)), 
                          transform=trans)
            stripes.append(s)
        return stripes

def plotDuffingPrediction(args, yPred, yTar, bidx=0, epoch=0):
    '''
    Plots the prediction of the model for the duffing equation in the space
    of the two state variables. A color map is used to denote time
    progression.
    Args:
        args (argparse): object with programs arguements
        yPred (torch.Tensor): [mb x t] tensor of a batch of model predictions
        yTar (torch.Tensor): [mb x t] tensor of a batch of targets
        bidx (int): integer specifying batch case to plot
    '''
    yPred = yPred.cpu()
    yTar = yTar.cpu()
    plt.close('all')

    # Use color map to indicate time
    cmaps = [plt.get_cmap("spring"), plt.get_cmap("summer")]
    predColors = cmaps[0](np.linspace(0,1,yPred.size(1)))
    tarColors = cmaps[1](np.linspace(0,1,yPred.size(1)))

    for i in range(0,yPred.size(1)-1):
        plt.plot(yPred[0,i:i+2,0], yPred[0,i:i+2,1], '--*', color=predColors[i])
        plt.plot(yTar[0,i:i+2,0], yTar[0,i:i+2,1], color=tarColors[i])

    plt.xlim([-2,2])
    plt.ylim([-2,2])
    cmap_handles = [Rectangle((0, 0), 1, 1) for _ in cmaps]
    handler_map = dict(zip(cmap_handles, 
                        [HandlerColormap(cm, num_stripes=8) for cm in cmaps]))

    # Create custom legend with color map rectangels
    plt.legend(handles=cmap_handles, labels=['Prediction','Target'], handler_map=handler_map, loc='upper right', framealpha=0.95)

    file_dir = args.pred_dir
    # If director does not exist create it
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_name = file_dir+"/{:s}Pred{:d}_{:d}".format(args.exp_name, bidx, epoch)
    plt.savefig(file_name+".png", bbox_inches='tight')

def plotEignValues(args, model, epoch=0):
    '''
    Plots the eigen values of the learned Koopman operator
    Args:
        args (argparse): object with programs arguements
        model (pytorch module): pytorch model with koopman operator kMatrix
    '''
    # Get koopman operator from model
    kMatrix = model.getKoopmanMatrix().detach().cpu().numpy()

    try:
        w, v = np.linalg.eig(kMatrix)
    except:
        print('issue computing eigs')
        return

    plt.close('all')
    plt.scatter(np.real(np.log(np.abs(w))), np.imag(w))

    file_dir = args.pred_dir+'/eig'
    # If director does not exist create it
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_name = file_dir+"/{:s}Eig_{:d}".format(args.exp_name, epoch)
    plt.savefig(file_name+".png", bbox_inches='tight')

def plotEignVectors(args, model, n=3, epoch=0):
    '''
    Plots the eigen vectors of the learned Koopman operator
    Args:
        args (argparse): object with programs arguements
        model (pytorch module): pytorch model with koopman operator kMatrix
    '''
    # Get koopman operator from model
    kMatrix = model.kMatrix.detach().cpu().numpy()
    w, v = np.linalg.eig(kMatrix)
    idx = w.argsort()[::-1]
    w = w[idx]
    v = v[:,idx]

    data = np.load('./duffing-test.npy')
    yPred = torch.Tensor(data[:,:,:-1])

    plt.close("all")
    fig, ax = plt.subplots(1, n, figsize=(4*n, 4))
    
    for i in range(n):
        ev_data = np.zeros(yPred.size(0))
        x_data = np.zeros(yPred.size(0))
        xdot_data = np.zeros(yPred.size(0))
        for j in range(yPred.size(0)):
            
            xin = yPred[j,:args.stride].to(args.device)
            xin = xin.view(-1).unsqueeze(0)

            g0,_= model(xin)
            g0 = g0.squeeze(0).detach().cpu().numpy()

            ev_data[j] = np.real(np.dot(g0,v[i]))
            x_data[j] = xin[0,-2].cpu().numpy()
            xdot_data[j] = xin[0,-1].cpu().numpy()


        ax[i].set_title('{:.02f} + {:.02f}i'.format(w[i].real, w[i].imag))
        ax[i].tricontourf(x_data, xdot_data, ev_data, cmap="RdBu_r")


    file_dir = args.pred_dir+'/eigVects'
    # If director does not exist create it
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_name = file_dir+"/{:s}EigVects_{:d}".format(args.exp_name, epoch)
    plt.savefig(file_name+".png", bbox_inches='tight')