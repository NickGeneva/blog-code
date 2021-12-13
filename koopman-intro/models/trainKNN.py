'''
Into to deep learning Koopman operators
===
Author: Nicholas Geneva (MIT Liscense)
url: https://nicholasgeneva.com/blog/
github: https://github.com/NickGeneva/blog-code
===
'''
import torch
import numpy as np
import torch.nn as nn
from utils.viz import plotDuffingPrediction

def trainModel(args, model, training_loader, optimizer, tback=None, epoch=0):
    '''
    Trains model for a single epoch
    '''
    loss_total = 0
    mseLoss = nn.MSELoss()
    for mbidx, (yInput0, yInput1) in enumerate(training_loader):
        
        batch_size = yInput0.size(0)
        xin0 = yInput0.view(batch_size, -1).to(args.device) # Time-step 

        # Model forward for both time-steps
        g0, xRec0 = model(xin0)
        loss = mseLoss(xin0, xRec0)

        g1Old = g0
        # Koopman transform
        for t0 in range(yInput1.shape[1]):
            xin0 = torch.cat([xin0[:,2:].detach(), yInput1[:,t0].to(args.device)], dim=1) # Next time-step
            g1, xRec1 = model(xin0)

            g1Pred = model.koopmanOperation(g0)
            xgRec1 = model.recover(g1Pred)

            # Multi-component loss enforcing reconstruction, linear dynamics, and koopman sparsity
            loss = loss + mseLoss(g1Pred, g1) + 10*mseLoss(xgRec1, xin0) + mseLoss(xRec1, xin0)\
               + 0.1*torch.mean(torch.abs(model.kMatrixDiag)) + 0.1*torch.mean(torch.abs(model.kMatrixUT))

            if(not tback is None and (t0+1)%tback == 0):
                loss_total = loss_total + loss.detach()
                # Backwards!
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                g0 = g1Pred.detach()
                xin0 = xin0.detach()
                loss = 0
            else:
                g0 = g1Pred

        if(tback is None or tback > yInput1.shape[1]):            
            loss_total = loss_total + loss.detach()
            # Backwards!
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            optimizer.zero_grad()
        
        loss = 0

    return loss_total

def testModel(args, model, testing_loader, epoch=0, plot=True):
    '''
    Tests model for a single epoch
    '''
    model.eval()

    test_loss = 0
    mseLoss = nn.MSELoss()
    for mbidx, (yInput0, yTarget0) in enumerate(testing_loader):
        
        batch_size = yInput0.size(0)
        xin0 = yInput0.view(batch_size, -1).to(args.device) # Testing starting input
        
        yTar = yTarget0.to(args.device)
        yPred = torch.zeros(yTar.size()).to(args.device)
        # Get initial observables
        g0, xRec0 = model(xin0)
        for i in range(yTar.size(1)):
            g0 = model.koopmanOperation(g0)
            yPred0 = model.recover(g0)
            # g0,_ = model(yPred0)

            yPred[:,i] = yPred0[:,-2:].detach()

        test_loss = test_loss + mseLoss(yTar, yPred)

        # Plot prediction versus target
        if(plot and mbidx == 0):
            plotDuffingPrediction(args, yPred, yTar, bidx=0, epoch=epoch)

    return test_loss
