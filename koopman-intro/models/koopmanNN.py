'''
Into to deep learning Koopman operators
===
Author: Nicholas Geneva (MIT Liscense)
url: https://nicholasgeneva.com/blog/
github: https://github.com/NickGeneva/blog-code
===
'''
import torch
import torch.nn as nn
from torch.autograd import Variable

class KoopmanNetwork(nn.Module):

    def __init__(self, indim, obsdim, xdim=10):
        super(KoopmanNetwork, self).__init__()
        self.observableNet = nn.Sequential(
            nn.Linear(indim, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, obsdim)
        )

        self.recoveryNet = nn.Sequential(
            nn.Linear(obsdim, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, indim)
        )
        # Learned koopman operator
        # Learns skew-symmetric matrix with a diagonal
        self.kMatrixDiag = nn.Parameter(torch.rand(obsdim))
        self.kMatrixUT = nn.Parameter(0.01*torch.randn(int(obsdim*(obsdim-1)/2)))
        
        # self.kMatrix = nn.Parameter(torch.rand(obsdim, obsdim))
        self.obsdim = obsdim
        print('Total number of parameters: {}'.format(self._num_parameters()))

    def forward(self, x):
        g = self.observableNet(x)
        x0 = self.recoveryNet(g)

        return g, x0

    def recover(self, g):
        x0 = self.recoveryNet(g)
        return x0

    def koopmanOperation(self, g):
        '''
        Applies the learned koopman operator on the given observables.
        Args:
            g (torch.Tensor): [b x  g] batch of observables, must match dim of koopman transform
        Returns:
            gnext (torch.Tensor): [b x g] predicted observables at the next time-step
        '''
        # assert g.size(-1) == self.kMatrix.size(0), 'Observables should have dim {}'.format(self.kMatrix.size(0))
        # Build Koopman matrix (skew-symmetric with diagonal)
        kMatrix = Variable(torch.Tensor(self.obsdim, self.obsdim)).to(self.kMatrixUT.device)

        utIdx = torch.triu_indices(self.obsdim, self.obsdim, offset=1)
        diagIdx = torch.stack([torch.arange(0,self.obsdim,dtype=torch.long).unsqueeze(0), \
            torch.arange(0,self.obsdim,dtype=torch.long).unsqueeze(0)], dim=0)
        kMatrix[utIdx[0], utIdx[1]] = self.kMatrixUT
        kMatrix[utIdx[1], utIdx[0]] = -self.kMatrixUT
        kMatrix[diagIdx[0], diagIdx[1]] = torch.nn.functional.relu(self.kMatrixDiag)

        gnext = torch.bmm(g.unsqueeze(1), kMatrix.expand(g.size(0), kMatrix.size(0), kMatrix.size(0)))
        return gnext.squeeze(1)

    def getKoopmanMatrix(self, requires_grad=False):
        '''
        Returns current Koopman operator
        '''
        kMatrix = Variable(torch.Tensor(self.obsdim, self.obsdim), requires_grad=requires_grad).to(self.kMatrixUT.device)

        utIdx = torch.triu_indices(self.obsdim, self.obsdim, offset=1)
        diagIdx = torch.stack([torch.arange(0,self.obsdim,dtype=torch.long).unsqueeze(0), \
            torch.arange(0,self.obsdim,dtype=torch.long).unsqueeze(0)], dim=0)
        kMatrix[utIdx[0], utIdx[1]] = self.kMatrixUT
        kMatrix[utIdx[1], utIdx[0]] = -self.kMatrixUT
        kMatrix[diagIdx[0], diagIdx[1]] = torch.nn.functional.relu(self.kMatrixDiag)

        return kMatrix

    def _num_parameters(self):
        count = 0
        for name, param in self.named_parameters():
            # print(name, param.numel())
            count += param.numel()
        return count
