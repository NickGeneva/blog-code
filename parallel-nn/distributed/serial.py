'''
Learning How To Train Neural Networks in Parallel (The Right Way)
===
Author: Nicholas Geneva (MIT Liscense)
url: https://nicholasgeneva.com/blog/
github: https://github.com/NickGeneva/blog-code
===
'''
import torch
from .dist import Distributed

class Serial(Distributed):
    """Place holder distributed comm class for serial training
    """
    def __init__(self):
        super().__init__()
        print("Warning: Using serial training")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @property
    def comm(self):
        return None

    @property
    def name(self):
        return f'{self.device}'

    def barrier(self):
        pass

    def copy_model(self,  *args, **kwargs):
        pass

    def average_gradients(self, *args, **kwargs):
        pass

    def comm_error(self, *args, **kwargs):
        return 0