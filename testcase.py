import numpy as np
from ctensor import Tensor
from ctensor.functional import conv2d, relu


D = Tensor.ones((4, 1, 199, 199))
W1 = Tensor.ones((1, 1, 3, 3))

I = conv2d(D, W1, padding=(1, 1))
I.mean().backward()
print(W1.grad)

D = Tensor.ones((1, 1, 19, 19))
W1 = Tensor.ones((1, 1, 3, 3))

I = conv2d(D, W1, padding=(1, 1))
I.mean().backward()
print(W1.grad)

D = Tensor.ones((1, 100))
W1 = Tensor.ones((100, 50))
(D @ W1).mean().backward()
print(W1.grad)

D = Tensor.ones((100, 100))
W1 = Tensor.ones((100, 50))
(D @ W1).mean().backward()
print(W1.grad)
