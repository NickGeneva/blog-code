'''
Learning How To Train Neural Networks in Parallel (The Right Way)
===
Author: Nicholas Geneva (MIT Liscense)
url: https://nicholasgeneva.com/blog/
github: https://github.com/NickGeneva/blog-code
===
conda install openmpi openmpi-mpicc
pip install mpi4py
'''
import os
import socket
import torch
import torch.distributed as torch_dist
from .dist import Distributed

try:
    from mpi4py import MPI
except:
    MPI = None

class Gloo(Distributed):
    """PyTorch Gloo Environment. Relevant docs:
    https://pytorch.org/docs/stable/distributed.html

    Args:
        addr (str, optional): Master IP address. Defaults to None.
        port (int, optional): Master port. Defaults to 29500.
    """
    def __init__(self, addr:str=None, port:int = 29500):
        
        if MPI == None:
            raise ModuleNotFoundError("Gloo distributed training needs mpi4py")


        # Global MPI environment
        self.mpi_comm = MPI.COMM_WORLD
        self.split_cuda_devices()

        # If no socket provided get default from rank 0 and broadcast
        # Can use 127.0.0.1 if on single node
        if addr is None:
            if self.rank == 0:
                addr = socket.gethostbyname(socket.gethostname())
                addr = self.mpi_comm.bcast(addr, root=0)
            else:
                addr = self.mpi_comm.bcast(None, root=0)

        print(f"Proc {self.rank}, Initializing Gloo enviroment of size {self.size}")
        print(f"Proc {self.rank}, Master address {addr}:{str(port)}")
        backend = "gloo"
        os.environ["MASTER_ADDR"] = addr
        os.environ['MASTER_PORT'] = str(port)
        os.environ['WORLD_SIZE'] = str(self.size)
        torch_dist.init_process_group(backend, rank=self.rank, world_size=self.size)

    def split_cuda_devices(self):
        """Splits cuda devices between MPI processes.
        """
        # Determine number of cuda devices on local node parallel environment
        if(torch.cuda.device_count() >= 1):
            n_gpu = torch.cuda.device_count()
            # NOT COMPLETELY SAFE, ASSUMES MPI HAS SEQUENTIALLY DISTRUBTED TASKS BETWEEN NODES
            # IDEALLY there should be a global comm on cuda assignments
            self.device = torch.device("cuda:{:d}".format(self.rank % n_gpu) if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"

    @property
    def size(self):
        return self.mpi_comm.Get_size()

    @property
    def rank(self):
        return self.mpi_comm.Get_rank()

    @property
    def comm(self):
        return self.mpi_comm

    @property
    def name(self):
        return f'Proc {self.rank}; {self.device}'

    def barrier(self):
        self.mpi_comm.Barrier()

    def copy_model(self, model, root:int = 0):
        """Copies model parameters from model on root process to all others

        Args:
            model (nn.Module): PyTorch model
            root (int, optional): MPI root id. Defaults to 0.
        """
        # Loop through model parameters
        for _, param in enumerate(model.parameters()):
            torch_dist.broadcast(param.data, src=root)


    def average_gradients(self, model):
        """ Averages the models gradients between all processes
        
        Args:
            model (nn.Module): PyTorch model
        """

        # Loop through model parameters
        for _, param in enumerate(model.parameters()):
            # If parameter has a gradient exchange
            if not param.grad is None:
                torch_dist.all_reduce(param.grad, op=torch_dist.ReduceOp.SUM)
                param.grad.data  = param.grad.data / self.size


    def comm_error(self, model) -> float:
        """Computes the means squared error between the models on 
        different devices. Used to check consistency.

        Args:
            model (nn.Module): PyTorch model
        Returns:
            (float): Mean squared error of model parameters
        """
        error = 0
        n = 0 # Number of parameters
        err_fct = torch.nn.MSELoss()

        # Loop through model parameters
        for _, param in enumerate(model.parameters()):
            # If parameter has a gradient exchange
            if not param.data is None:
                
                param_sum = param.data.clone()
                torch_dist.all_reduce(param_sum, op=torch_dist.ReduceOp.SUM)

                param_sum =  param_sum / self.size
                error += err_fct(param.data.cpu(), param_sum.cpu())
                n += 1

        return error/n