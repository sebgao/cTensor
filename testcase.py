import numpy as np
from ctensor import Tensor, Operator

class Mul(Operator):
    def __init__(self, coff=1):
        self.coff = coff

    def forward(self, x, y):
        return self.coff*x*y

def mul(x, y):
    return Mul(3)(x, y)

a = Tensor(np.array([1, 2]))
b = Tensor(np.array([2, 3]))
r = a.view(1, 1, 2) @ b.view(1, 2, 1)
print(r)
r.backward()
print(a.grad)
