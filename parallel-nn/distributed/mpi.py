'''
Learning How To Train Neural Networks in Parallel (The Right Way)
===
Author: Nicholas Geneva (MIT Liscense)
url: https://nicholasgeneva.com/blog/
github: https://github.com/NickGeneva/blog-code
===
If you dont have MPI installed already:
conda install openmpi openmpi-mpicc
pip install mpi4py


Otherwise you should build MPI4Py from source:
Example load modules: mpich/3.3/gcc/8.3.0

Install some needed libraries into conda env:
conda install -c conda-forge libgfortran-ng
conda install -c anaconda gcc_linux-64

Update config in source folder of MPI4Py:
[mpich]
mpi_dir              = <FOLDER TO MPI (use `which mpirun`)>
mpicc                = %(mpi_dir)s/bin/mpicc
mpicxx               = %(mpi_dir)s/bin/mpicxx

Build:
python setup.py build --mpi=mpich
python setup.py install
'''
import torch
from .dist import Distributed
try:
    from mpi4py import MPI
except:
    MPI = None

class Mpi(Distributed):
    """MPI Environment using mpi4py library
    """
    def __init__(self):
        
        if MPI == None:
            raise ModuleNotFoundError("MPI distributed training needs mpi4py")

        # Global MPI environment
        self.mpi_comm = MPI.COMM_WORLD
        self.split_cuda_devices()

        print(f"Proc {self.rank}, Initializing MPI enviroment of size {self.size}")

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
        return self.comm.Get_size()

    @property
    def rank(self):
        return self.comm.Get_rank()

    @property
    def comm(self):
        return self.mpi_comm

    @property
    def name(self):
        return f'Proc {self.rank}; {self.device}'

    def barrier(self):
        self.mpi_comm.Barrier()

    def copy_model(self, model, root:int = 0, device = None):
        """Copies model parameters from model on root process to all others

        Args:
            model (nn.Module): PyTorch model
            root (int, optional): MPI root id. Defaults to 0.
            device (torch.device, optional): PyTorch device model to put params on. Defaults to None.
        """
        if device is None:
            device = self.device

        # Loop through model parameters
        for _, param in enumerate(model.parameters()):
            if self.rank == root:
                param_cpu = param.data.cpu()
            else:
                param_cpu = None
            # Uses pickle-based comm which is appearently near C-speed
            # Can use capital method calls if you want to specify data-type
            param_cpu = self.mpi_comm.bcast(param_cpu, root=root)
            param.data  = param_cpu.to(device)


    def average_gradients(self, model, device=None):
        """ Averages the models gradients between all processes
        
        Args:
            model (nn.Module): PyTorch model
            device (torch.device, optional): PyTorch device. Defaults to None.
        """
        if device is None:
            device = self.device

        # Loop through model parameters
        for _, param in enumerate(model.parameters()):
            # If parameter has a gradient exchange
            if not param.grad is None:
                param_cpu = param.grad.data.cpu()
                param_sum = self.comm.allreduce(param_cpu, op=MPI.SUM)
                param.grad.data  = param_sum.to(device) / self.size


    def comm_error(self, model, device=None) -> float:
        """Computes the means squared error between the models on 
        different devices. Used to check consistency.

        Args:
            model (nn.Module): PyTorch model
            device (torch.device, optional): PyTorch device. Defaults to None.
            
        Returns:
            (float): Mean squared error of model parameters
        """
        if device is None:
            device = self.device

        error = 0
        n = 0 # Number of parameters
        err_fct = torch.nn.MSELoss()

        # Loop through model parameters
        for _, param in enumerate(model.parameters()):
            # If parameter has a gradient exchange
            if not param.data is None:

                param_cpu = param.data.cpu()
                param_sum = self.comm.allreduce(param_cpu, op=MPI.SUM)

                param_sum =  param_sum / self.size
                error += err_fct(param.data.cpu(), param_sum.cpu())
                n += 1

        return error/n