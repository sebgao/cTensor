import numpy as np
from .tensor import Tensor
from . import functional as F

class Parameter(Tensor):
    def __init__(self, data):
        if isinstance(data, np.ndarray):
            Tensor.__init__(self, data)
        else:
            Tensor.__init__(self, data.data)

class Module:
    def parameters(self):
        params = [getattr(self, i) for i in dir(self) if isinstance(
            getattr(self, i), Parameter)]
        submodules = [getattr(self, i) for i in dir(
            self) if isinstance(getattr(self, i), Module)]
        for p in params:
            yield p
        for sm in submodules:
            yield from sm.parameters()

    def __call__(self, *args):
        return self.forward(*args)

class Sequential(Module):
    def __init__(self, *args):
        self.seqential = args
    
    def forward(self, x):
        for s in self.seqential:
            x = s(x)
        return x
    
    def parameters(self):
        for sm in self.seqential:
            yield from sm.parameters()

class Linear(Module):
    def __init__(self, in_channels, out_channels, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w = Parameter(np.random.randn(
            in_channels, out_channels)/((in_channels*out_channels)))
        if bias:
            self.bias = Parameter(np.zeros(((1, out_channels))))
        else:
            self.bias = 0
    
    def forward(self, x):
        return x @ self.w + self.bias

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=True):
        param = (in_channels, ) + (out_channels, ) + kernel_size
        self.w = Parameter(
            np.random.randn(*param)/(in_channels*(kernel_size[0]*kernel_size[1])**0.5)
            )
        if bias:
            self.bias = Parameter(np.zeros((1, out_channels, 1, 1)))
        else:
            self.bias = 0
        self.padding = padding
        self.stride = stride
    
    def forward(self, x):
        return F.conv2d(x, self.w, self.padding, self.stride) + self.bias

class ReLU(Module):
    def forward(self, x):
        return F.relu(x)


class LeakyReLU(Module):
    def forward(self, x):
        return F.leaky_relu(x)

class Sigmoid(Module):
    def forward(self, x):
        return F.sigmoid(x)
