import numpy as np
from .tensor import *

class Optimizer:
    def __init__(self, parameters):
        assert parameters, 'Your parameters?'
        self.parameters = list(parameters)

    def zero_grad(self):
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)

    def step(self, lr=0.01):
        assert False, 'Optimizer class is virtual'


class SGD(Optimizer):
    def step(self, lr=0.01):
        for p in self.parameters:
            p.data -= lr*p.grad


class Adam(Optimizer):
    def __init__(self, parameters, beta1=0.9, beta2=0.99, eta=1e-6):
        super(Adam, self).__init__(parameters)
        self.m = []
        self.v = []
        for p in self.parameters:
            self.m.append(np.zeros_like(p.grad))
            self.v.append(np.zeros_like(p.grad))

        self.beta1 = beta1
        self.beta2 = beta2
        self.eta = eta

    def step(self, lr=0.01):
        '''
        Adam optimizer stepping.
        We ignore the unbiasing process as it hurts efficiency.
        '''

        beta1 = self.beta1
        beta2 = self.beta2
        for idx, p in enumerate(self.parameters):
            m = self.m[idx]
            v = self.v[idx]
            grad = p.grad
            m[...] = beta1*m + (1-beta1)*grad
            v[...] = beta2*v + (1-beta2)*(grad**2)
            # ignoring unbiasing
            p.data -= lr*m/(np.sqrt(v)+self.eta)

