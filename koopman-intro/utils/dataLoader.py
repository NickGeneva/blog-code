'''
Into to deep learning Koopman operators
===
Author: Nicholas Geneva (MIT Liscense)
url: https://nicholasgeneva.com/blog/
github: https://github.com/NickGeneva/blog-code
===
'''
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def loadData(ntrain, ntest, stride, file_name, batch_size=16, target_steps=20, shuffle=True):
    '''
    Loads time-series data and creates training/testing loaders
    '''
    data = np.load(file_name)

    idx = np.arange(0,data.shape[0],1)
    np.random.shuffle(idx)
    training_data = data[idx[:ntrain],:,:-1]
    testing_data = data[idx[ntrain:ntrain+ntest],:,:-1]

    
    # Split time-series into strides
    endIdx = training_data.shape[1] - target_steps - stride
    input_data = np.concatenate([training_data[:,i:i+stride] for i in range(0,endIdx,target_steps)], axis=0)
    target_data = np.concatenate([training_data[:,i:i+target_steps] for i in range(stride-1, training_data.shape[1] - target_steps + 1, target_steps+stride)], axis=0)
    input_data = training_data[:,:stride]
    target_data = training_data[:,stride:]
    dataset = TensorDataset(torch.Tensor(input_data), torch.Tensor(target_data))
    training_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

    dataset = TensorDataset(torch.Tensor(testing_data[:,:stride]), torch.Tensor(testing_data[:,stride:]))
    testing_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return training_loader, testing_loader

def loadTestData(ntest, stride, file_name, batch_size=16, shuffle=False):
    '''
    Loads time-series data and creates training/testing loaders
    '''
    data = np.load(file_name)

    testing_data = data[:ntest,:,:-1]

    # Make test data-set with (initial state and then remaining target values in time-series)
    dataset = TensorDataset(torch.Tensor(testing_data[:,:stride]), torch.Tensor(testing_data[:,stride:]))
    testing_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    return testing_loader