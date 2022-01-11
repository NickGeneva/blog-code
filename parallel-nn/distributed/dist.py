'''
Learning How To Train Neural Networks in Parallel (The Right Way)
===
Author: Nicholas Geneva (MIT Liscense)
url: https://nicholasgeneva.com/blog/
github: https://github.com/NickGeneva/blog-code
===
'''
from abc import abstractmethod

class Distributed(object):
    """Parent class for  distributed comm methods
    """
    @property
    def size(self):
        return 1

    @property
    def rank(self):
        return 0

    @property
    def comm(self):
        return None

    @property
    def name(self):
        return ""

    @abstractmethod
    def barrier(self):
        raise NotImplementedError("barrier function should be overloaded")

    @abstractmethod
    def copy_model(self, *args, **kwargs):
        raise NotImplementedError("copy_model function should be overloaded")

    @abstractmethod
    def average_gradients(self, *args, **kwargs):
        raise NotImplementedError("average_gradients function should be overloaded")

    @abstractmethod
    def comm_error(self, *args, **kwargs):
        raise NotImplementedError("comm_error function should be overloaded")