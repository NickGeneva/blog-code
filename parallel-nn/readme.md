# How to Train Neural Networks in Parallel (From Scratch)

## Installing Dependencies
### Conda install MPI:
```
conda install openmpi openmpi-mpicc
pip install mpi4py
```
Depending on the machine you are running you may have to install mpi4py installing from source using pre-existing MPI libaries.

### Conda install NCCL
```
conda install -c conda-forge cupy cudnn cutensor nccl
```
### Conda install Gloo/NCCL (PyTorch Cuda 10.2)
```
conda install pytorch cudatoolkit=10.2 -c pytorch
```
Will also need matplotlib for running these examples.

For MNIST example download the [dataset](http://yann.lecun.com/exdb/mnist/).

## Execution Examples
**Note**: MPI commands may depend on which MPI your are using. Provided are using mpich.
### Franke's Equation
Running MPI with 4 tasks:
```
mpirun -np 4 python main_franke.py --comm mpi
```
Running NCCL with 4 tasks:
```
mpirun -np 4 python main_franke.py --comm nccl
```
Running Gloo with 4 tasks:
```
mpirun -np 4 python main_franke.py --comm gloo
```
Running NCCL (PyTorch) with 4 tasks:
```
mpirun -np 4 python main_franke.py --comm ncclp
```