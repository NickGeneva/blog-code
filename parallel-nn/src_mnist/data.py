'''
How to Train Neural Networks in Parallel (From Scratch)
===
Author: Nicholas Geneva (MIT Liscense)
url: https://nicholasgeneva.com/blog/
github: https://github.com/NickGeneva/blog-code
===
'''
import torch
import gzip
import numpy as np
from typing import Tuple, Dict, List
from torch.utils.data.dataset import Dataset
from distributed.dist import Distributed
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler
from pathlib import Path

def create_training_dataloader(
    ndata: int,
    batch_size: int,
    dcomm: Distributed,
    data_path:Path = Path('.'),
    num_workers: int = 1,
    seed: int = 0
):
    image_file = data_path / "train-images-idx3-ubyte.gz"
    label_file = data_path / "train-labels-idx1-ubyte.gz"
    dataset = MNISTDataset(image_file, label_file, ndata, dcomm.rank, dcomm.size, seed)
    sampler = RandomSampler(dataset)

    batch_size = batch_size // dcomm.size
    if batch_size > len(dataset):
        batch_size = len(dataset)

    data_loader = DataLoader(dataset, 
        sampler = sampler,
        batch_size = batch_size,
        drop_last = True,
        num_workers = num_workers)

    return data_loader

def create_testing_dataloader(
    ndata: int,
    batch_size: int,
    dcomm: Distributed = None,
    data_path:Path = Path('.'),
    num_workers: int = 1,
    seed: int = 12345
):
    image_file = data_path / "t10k-images-idx3-ubyte.gz"
    label_file = data_path / "t10k-labels-idx1-ubyte.gz"
    if dcomm is None:
        dataset = MNISTDataset(image_file, label_file, ndata, seed=seed)
    else:
        dataset = MNISTDataset(image_file, label_file, ndata, dcomm.rank, dcomm.size, seed)
        batch_size = batch_size // dcomm.size


    sampler = SequentialSampler(dataset)

    if batch_size > len(dataset):
        batch_size = len(dataset)

    data_loader = DataLoader(dataset, 
        sampler = sampler,
        batch_size = batch_size,
        drop_last = False,
        num_workers = num_workers)

    return data_loader


class MNISTDataset(TensorDataset):

    def __init__(
        self,
        image_file_name: str,
        label_file_name: str,
        ndata: int,
        rank: int = 0, 
        size: int = 1, 
        seed: int = 0
    ):
        assert Path(image_file_name).is_file(), "Provided dataset file cannot be found!"
        assert Path(label_file_name).is_file(), "Provided label dataset file cannot be found!"

        self.seed = seed
        self.load_dataset(image_file_name, label_file_name, ndata, rank, size)

        super().__init__(self.examples, self.labels)
        
    def load_dataset(self, 
        image_file_name: str,
        label_file_name: str,
        ndata: int, 
        rank: int, 
        size: int, 
        seed:int = None
    ):
        """Reads MNIST data from file

        Args:
            ndata (int): Total number of data in dataset
            rank (int): Distributed training process rank
            size (int): Size of distrubted process communication world
            seed (int, optional): Random seed. Defaults to None.
        """
        if not seed:
            seed = self.seed

        # Load entire data-set
        # Typically you would use built in PyTorch version but we want to split data
        f_image = gzip.open(image_file_name, 'r')
        f_label = gzip.open(label_file_name, 'r')
        f_image.read(16)
        f_label.read(8)
        image_size = 28

        # Now we split data evenly based on MPI rank
        sidx = rank * (ndata // size)
        eidx = (rank + 1) * (ndata // size)

        # Skip these images
        f_image.read(image_size * image_size * sidx)
        # Read images for this process
        buf = f_image.read(image_size * image_size * (eidx - sidx))
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(ndata // size, image_size, image_size) / 255.
        self.examples = torch.from_numpy(data)

        f_label.read(sidx)
        buf = f_label.read(eidx - sidx)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.uint8)
        self.labels = torch.from_numpy(data)

    def __len__(self):
        return self.examples.shape[0]