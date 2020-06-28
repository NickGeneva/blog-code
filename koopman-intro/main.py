'''
Into to deep learning Koopman operators
===
Author: Nicholas Geneva (MIT Liscense)
url: https://nicholasgeneva.com/blog/
github: https://github.com/AbsoluteStratos/blog-code
===
'''
import numpy as np
import random
import argparse
import os, errno, copy, json
from args import Parser

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, CyclicLR
from utils.dataLoader import loadData

from models.koopmanNN import KoopmanNetwork
from models.trainKNN import trainModel, testModel

from utils.viz import plotDuffingPrediction, plotEignValues, plotEignVectors
    
if __name__ == '__main__':

    args = Parser().parse()    
    if(torch.cuda.is_available()):
        use_cuda = "cuda"
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Torch device:{}".format(args.device))
    scheduler = None
    # Set up data loaders
    training_loader, testing_loader = loadData(args.ntrain, args.ntest, args.stride, './duffing.npy', batch_size=args.batch_size, target_steps=100)
    
    # Init model and optimizer 
    model = KoopmanNetwork(2*args.stride, 50).to(args.device)

    parameters = [{'params': [model.kMatrixDiag, model.kMatrixUT], 'lr': 1e-2},
                {'params': [model.observableNet.parameters(), model.recoveryNet.parameters()]}  ]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-7)
    scheduler = ExponentialLR(optimizer, gamma=0.995)

    for epoch in range(1,args.epochs+1):
        # Evaluate 1 epoch of the model
        loss = trainModel(args, model, training_loader, optimizer, epoch=epoch)

        print('Epoch {:d}: Training Loss {:.02f}'.format(epoch, loss))

        if(not scheduler is None):
            scheduler.step()
            if(epoch%10 == 0):
                for param_group in optimizer.param_groups:
                    print('Epoch {:d}; Learning-rate: {:0.05f}'.format(epoch, param_group['lr']))

        # Test model
        if(epoch%5 == 0 or epoch == 1):
            with torch.no_grad():
                test_loss = testModel(args, model, testing_loader, epoch=epoch)
                print('Epoch {:d}: Testing Loss {:.02f}'.format(epoch, test_loss))
                plotEignValues(args, model, epoch)

        if(epoch%10 == 0 or epoch == 1):
            # Save model
            torch.save(model.state_dict(), args.ckpt_dir+'/torchModel{:d}.pth'.format(epoch))
        
