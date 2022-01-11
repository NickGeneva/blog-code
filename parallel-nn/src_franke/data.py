'''
How to Train Neural Networks in Parallel (From Scratch)
===
Author: Nicholas Geneva (MIT Liscense)
url: https://nicholasgeneva.com/blog/
github: https://github.com/NickGeneva/blog-code
===
'''
import torch
from typing import Tuple, Dict, List
from torch.utils.data.dataset import Dataset
from distributed.dist import Distributed
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler


def create_training_dataloader(
    ndata: int,
    batch_size: int,
    dcomm: Distributed,
    num_workers: int = 1,
    seed: int = 0
):

    dataset = FrankeDataset(ndata, dcomm.rank, dcomm.size, seed=seed)
    collator = FrankeDataCollator()
    sampler = RandomSampler(dataset)

    batch_size = batch_size // dcomm.size
    if batch_size > len(dataset):
        batch_size = len(dataset)

    data_loader = DataLoader(dataset, 
        sampler = sampler,
        batch_size = batch_size,
        collate_fn = collator, 
        drop_last = True,
        num_workers = num_workers)

    return data_loader

def create_testing_dataloader(
    ndata: int,
    batch_size: int,
    dcomm: Distributed = None,
    num_workers: int = 1,
    seed: int = 12345
):
    if dcomm is None:
        dataset = FrankeDataset(ndata, seed=seed)
    else:
        dataset = FrankeDataset(ndata, dcomm.rank, dcomm.size, seed=seed)
        batch_size = batch_size // dcomm.size

    collator = FrankeDataCollator()
    sampler = SequentialSampler(dataset)

    if batch_size > len(dataset):
        batch_size = len(dataset)

    data_loader = DataLoader(dataset, 
        sampler = sampler,
        batch_size = batch_size,
        collate_fn = collator, 
        drop_last = False,
        num_workers = num_workers)

    return data_loader


class FrankeDataset(Dataset):

    def __init__(
        self,
        ndata: int,
        rank: int = 0, 
        size: int = 1, 
        seed: int = 0
    ):
        super().__init__()
        
        self.seed = seed
        self.load_dataset(ndata, rank, size)

    def franke(self, x1: torch.Tensor, x2: torch.Tensor):
        """Calculates 2D franke function
        https://www.sfu.ca/~ssurjano/franke2d.html

        Args:
            x1 (torch.Tensor): x coordinate tensor
            x2 (torch.Tensor): y coordinate tensor

        Returns:
            torch.Tensor: f(x,y)
        """
        f = 0.75*torch.exp(-(9*x1-2)**2/4.0-(9*x2-2)**2/4.0) + 0.75*torch.exp(-(9*x1+1)**2/49.0-(9*x2+1)**2/10) \
            + 0.5*torch.exp(-(9*x1-7)**2/4.0-(9*x2-3)**2/4.0) - 0.2*torch.exp(-(9*x1-4)**2-(9*x2-7)**2)
        return f
        
    def load_dataset(self, ndata: int, crank: int, csize: int, seed:int = None):
        """Generates data-set for each process

        Args:
            ndata (int): Total number of data in dataset
            crank (int): Distributed training process rank
            csize (int): Size of distrubted process communication world
            seed (int, optional): Random seed. Defaults to None.
        """
        if not seed:
            seed = self.seed
        # Load (generate) entire data-set
        g_cpu = torch.Generator(device='cpu').manual_seed(seed)
        x = 2*torch.rand(ndata, 2, generator = g_cpu) - 1
        y = self.franke(x[:,0], x[:,1])

        # Now we split data evenly based on MPI rank
        sidx = crank * (ndata // csize)
        eidx = (crank + 1) * (ndata // csize)

        self.x = x[sidx:eidx]
        self.y = y[sidx:eidx]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i) -> Tuple[torch.Tensor]:
        return self.x[i], self.y[i]

class FrankeDataCollator:
    """
    Data collator
    """
    # Default collator
    def __call__(self, examples:List) -> Dict[str, torch.Tensor]:

        # Combine training examples
        inputs_ = []
        targets_ = []
        for _, example in enumerate(examples):
            inputs_.append( example[0] )
            targets_.append( example[1] )

        input_tensor = torch.stack(inputs_, dim=0) # [N, 2]
        target_tensor = torch.stack(targets_, dim=0).unsqueeze(-1) # [N, 1]

        return { "input": input_tensor, "target": target_tensor}

