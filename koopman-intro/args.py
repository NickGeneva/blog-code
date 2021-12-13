'''
Into to deep learning Koopman operators
===
Author: Nicholas Geneva (MIT Liscense)
url: https://nicholasgeneva.com/blog/
github: https://github.com/NickGeneva/blog-code
===
'''
import numpy as np
import random
import argparse
import os, errno, copy, json

import torch

class Parser(argparse.ArgumentParser):
    def __init__(self):
        super(Parser, self).__init__(description='Read')
        self.add_argument('--exp-dir', type=str, default="./koopman", help='directory to save experiments')
        self.add_argument('--exp-name', type=str, default="duffing", help='experiment name')
        self.add_argument('--model', type=str, default="fcnn", choices=['fcnn'], help='experiment name')

        # data
        self.add_argument('--ntrain', type=int, default=200, help="number of training data")
        self.add_argument('--ntest', type=int, default=5, help="number of training data")
        self.add_argument('--stride', type=int, default=10, help="number of time-steps as encoder input")
        self.add_argument('--batch-size', type=int, default=16, help='batch size for training')

        # training
        self.add_argument('--epoch-start', type=int, default=0, help='epoch to start at, will load pre-trained network')
        self.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
        self.add_argument('--lr', type=float, default=0.001, help='ADAM learning rate')
        self.add_argument('--seed', type=int, default=12345, help='manual seed used in PyTorch and Numpy')
        
        # logging
        self.add_argument('--plot-freq', type=int, default=25, help='how many epochs to wait before plotting test output')
        self.add_argument('--test-freq', type=int, default=5, help='how many epochs to test the model')
        self.add_argument('--ckpt-freq', type=int, default=5, help='how many epochs to wait before saving model')
        self.add_argument('--notes', type=str, default='')

    def mkdirs(self, *directories):
        '''
        Makes a directory if it does not exist
        '''
        for directory in list(directories):
            try:
                os.makedirs(directory)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

    def parse(self, dirs=True):
        '''
        Parse program arguements
        Args:
            dirs (boolean): True to make file directories for predictions and models
        '''
        args = self.parse_args()
        args.run_dir = args.exp_dir + '/' + '{}'.format(args.exp_name) \
            + '/{}_ntrain{}_batch{}_{}'.format(args.model, args.ntrain, args.batch_size, args.notes)

        args.ckpt_dir = args.run_dir + '/checkpoints'
        args.pred_dir = args.run_dir + "/predictions"
        if(dirs):
            self.mkdirs(args.run_dir, args.ckpt_dir, args.pred_dir)

        # Set random seed
        if args.seed is None:
            args.seed = random.randint(1, 10000)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(seed=args.seed)

        if dirs:
            with open(args.run_dir + "/args.json", 'w') as args_file:
                json.dump(vars(args), args_file, indent=4)

        return args