import numpy as np
from ctensor import Tensor
from ctensor.functional import conv2d, relu


D = Tensor.zeros((4, 1, 199, 199)) + 1
W1 = Tensor.zeros((1, 1, 3, 3)) + 1

I = conv2d(D, W1, padding=(1, 1))
I.mean().backward()
print(W1.grad)

D = Tensor.zeros((1, 1, 19, 19)) + 1
W1 = Tensor.zeros((1, 1, 3, 3)) + 1

I = conv2d(D, W1, padding=(1, 1))
I.mean().backward()
print(W1.grad)

D = Tensor.zeros((1, 100)) + 1
W1 = Tensor.zeros((100, 50)) + 1
(D @ W1).mean().backward()
print(W1.grad)

D = Tensor.zeros((100, 100)) + 1
W1 = Tensor.zeros((100, 50)) + 1
(D @ W1).mean().backward()
print(W1.grad)
