'''
How to Train Neural Networks in Parallel (From Scratch)
===
Author: Nicholas Geneva (MIT Liscense)
url: https://nicholasgeneva.com/blog/
github: https://github.com/NickGeneva/blog-code
===
'''
import argparse
import numpy as np
import torch
import random
from pathlib import Path

class Parser(argparse.ArgumentParser):

    def __init__(self):
        super(Parser, self).__init__(description='Read')
        self.add_argument('--comm', type=str, default="serial", choices=['serial', 'mpi', 'nccl', 'ncclp', 'gloo'], help='experiment name')
        # data
        self.add_argument('--ntrain', type=int, default=10000, help="number of training data")
        self.add_argument('--ntest', type=int, default=1000, help="number of training data")
        self.add_argument('--train-batch-size', type=int, default=256, help='batch size for training')
        self.add_argument('--test-batch-size', type=int, default=64, help='batch size for testing')

        # training
        self.add_argument('--epoch-start', type=int, default=0, help='epoch to start at, will load pre-trained network')
        self.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
        self.add_argument('--lr', type=float, default=0.001, help='ADAM learning rate')
        self.add_argument('--seed', type=int, default=12345, help='manual seed used in PyTorch and Numpy')
        
        # logging
        self.add_argument('--plot-freq', type=int, default=25, help='how many epochs to wait before plotting test output')
        self.add_argument('--test-freq', type=int, default=5, help='how many epochs to test the model')
        self.add_argument('--ckpt-freq', type=int, default=25, help='how many epochs to wait before saving the model')

    def parse(self):
        """
        Parse program arguements
        """
        args = self.parse_args()
        args.run_dir = Path('./mnist_outputs') / '{}'.format(args.comm) \
            / 'ntrain{}_batch{}'.format(args.ntrain, args.train_batch_size)

        args.ckpt_dir = args.run_dir / "checkpoints"
        args.pred_dir = args.run_dir / "predictions"
        for path in (args.run_dir, args.ckpt_dir, args.pred_dir):
            Path(path).mkdir(parents=True, exist_ok=True)

        # Set random seed
        if args.seed is None:
            args.seed = random.randint(1, 10000)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(seed=args.seed)

        return args