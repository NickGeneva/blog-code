'''
Learning How To Train Neural Networks in Parallel (The Right Way)
===
Author: Nicholas Geneva (MIT Liscense)
url: https://nicholasgeneva.com/blog/
github: https://github.com/NickGeneva/blog-code
===
conda install -c conda-forge cupy cudnn cutensor nccl
'''
import torch
import os
from .dist import Distributed

try:
    import cupy as cp
    from cupy.cuda import nccl
    from mpi4py import MPI
except:
    MPI = None

class Nccl(Distributed):
    """NCCL distributed training class. MPI is used to init the NCCL
    communicators using the mpi rank id. We will define 1 NCCL communicator
    per MPI task.

    Good example of available NCCL comm https://github.com/cupy/cupy/issues/2840
    """
    def __init__(self, debug:str = "WARN"):
        if MPI is None:
            raise ModuleNotFoundError("NCCL requires cupy with nccl and mpi4py installed.")

        self.mpi_comm = MPI.COMM_WORLD
        # Create NCCL communication ID for all processes
        nccl_comm_id = None
        if self.mpi_comm.Get_rank() == 0:
            nccl_comm_id = nccl.get_unique_id()
        nccl_comm_id = self.mpi_comm.bcast(nccl_comm_id, root=0)
        # Set the active CUDA device for this proc
        self.set_cuda_devices()

        print(f"Proc {self.rank}, Initializing NCCL enviroment of size {self.size}")
        # Create NCCL communicator (nRanks, id, myRank)
        os.environ["NCCL_DEBUG"] = debug
        self.nccl_comm = nccl.NcclCommunicator(self.size, nccl_comm_id, self.rank)

    def set_cuda_devices(self):
        """Splits cuda devices between processes.
        """
        # Determine number of cuda devices on local node parallel environment
        if(torch.cuda.device_count() >= 1 and torch.cuda.is_available()):
            n_gpu = torch.cuda.device_count()
            gpu_id = self.rank % n_gpu
            # NOT COMPLETELY SAFE, ASSUMES MPI HAS SEQUENTIALLY DISTRUBTED TASKS BETWEEN NODES
            # IDEALLY there should be a global comm on cuda assignments
            self.device = torch.device("cuda:{:d}".format(gpu_id))
            # Set NCCL device
            assert gpu_id < cp.cuda.runtime.getDeviceCount(), \
                f"Cuda id {gpu_id} is larger than device count {cp.cuda.runtime.getDeviceCount()}"
           
            device = cp.cuda.Device(gpu_id)
            cp.cuda.runtime.setDevice(device)
        else:
            raise Exception("Cuda devices not found to use with NCCL")

    @property
    def size(self):
        return self.mpi_comm.Get_size()

    @property
    def rank(self):
        return self.mpi_comm.Get_rank()

    @property
    def comm(self):
        return self.nccl_comm

    @property
    def name(self):
        return f'Proc {self.rank}; {self.device}'

    def barrier(self):
        # Use MPI here to sync tasks
        self.mpi_comm.Barrier()

    def get_NCCL_dtype(self, dtype):
        """Converts pytorch tensor datatype into NCCL datatype

        Args:
            dtype (torch.dtype): Tensor data type

        Raises:
            ValueError: Unsupported datatype

        Returns:
            nccl.DATA_TYPE: NCCL datatype value
        """
        # https://github.com/cupy/cupy/blob/master/cupy_backends/cuda/libs/nccl.pxd
        if dtype == cp.int8 or dtype == torch.int8:
            return nccl.NCCL_CHAR
        elif dtype == cp.uint8 or dtype == torch.uint8:
            return nccl.NCCL_UINT8
        elif dtype == cp.int32 or dtype == torch.int32:
            return nccl.NCCL_INT32
        elif dtype == cp.int64 or dtype == torch.int64:
            return nccl.NCCL_INT64
        elif dtype == cp.complex64 or dtype == torch.complex64:
            return nccl.NCCL_FLOAT32
        elif dtype == cp.complex128 or dtype == torch.complex128:
            return nccl.NCCL_FLOAT64
        elif dtype == cp.float32 or dtype == torch.float32:
            return nccl.NCCL_FLOAT32
        elif dtype == cp.float64 or dtype == torch.float64:
            return nccl.NCCL_FLOAT64
        else:
            raise ValueError("This dtype is not supported by NCCL.")

    def copy_model(self, model, root:int = 0, device = None, stream = None):
        """Copies parameters from model on root process to all others

        Args:
            model (nn.Module): PyTorch model
            root (int, optional): MPI root id. Defaults to 0.
            device (torch.device, optional): PyTorch device model to put params on. Defaults to None.
            stream (cuda.stream, optional): cuda stream to queue broadcast op. Defaults to None.
        """
        if device is None:
            device = self.device

        if stream is None:
            stream = cp.cuda.Stream.null.ptr
        else:
            stream = stream.ptr

        # Loop through model parameters
        for _, param in enumerate(model.parameters()):

            dtype = self.get_NCCL_dtype(param.data.dtype)
            size = int(torch.numel(param.data))
            root = 0

            self.nccl_comm.bcast(param.data_ptr(), size, dtype, root, stream)

    def average_gradients(self, model, device=None, stream = None):
        """ Averages the models gradients between all processes
        
        Args:
            model (nn.Module): PyTorch model
            device (torch.device, optional): PyTorch device. Defaults to None.
        """
        if device is None:
            device = self.device

        if stream is None:
            stream = cp.cuda.Stream.null.ptr
        else:
            stream = stream.ptr

        # Loop through model parameters
        for _, param in enumerate(model.parameters()):
            # If parameter has a gradient exchange
            if not param.grad is None:

                dtype = self.get_NCCL_dtype(param.grad.dtype)
                size = torch.numel(param.grad)
                recv_buff = torch.zeros_like(param.grad)

                # For ops see: https://github.com/cupy/cupy/blob/master/cupy_backends/cuda/libs/nccl.pxd
                self.nccl_comm.allReduce(param.grad.data_ptr(), recv_buff.data_ptr(), size, dtype, nccl.NCCL_SUM, stream)
                param.grad = recv_buff / self.size

    def comm_error(self, model, device=None, stream = None) -> float:
        """Computes the means squared error between the models on 
        different devices. Used to check consistency.

        Args:
            model (nn.Module): PyTorch model
            device (torch.device, optional): PyTorch device. Defaults to None.
            stream (cp.cuda.Stream.ptr, optional): cuda stream point. Defaults to None.
        
        Returns:
            (float): Mean squared error of model parameters
        """
        if device is None:
            device = self.device

        if stream is None:
            stream = cp.cuda.Stream.null.ptr
        else:
            stream = stream.ptr

        error = 0
        n = 0 # Number of parameters
        err_fct = torch.nn.MSELoss()

        # Loop through model parameters
        for _, param in enumerate(model.parameters()):
            # If parameter has a gradient exchange
            if not param.data is None:

                dtype = self.get_NCCL_dtype(param.data.dtype)
                size = torch.numel(param.data)
                recv_buff = torch.zeros_like(param.data)

                # For ops see: https://github.com/cupy/cupy/blob/master/cupy_backends/cuda/libs/nccl.pxd
                self.nccl_comm.allReduce(param.data.data_ptr(), recv_buff.data_ptr(), size, dtype, nccl.NCCL_SUM, stream)
                recv_buff =  recv_buff / self.size

                error += err_fct(param.data.cpu(), recv_buff.cpu())
                n += 1

        return error/n