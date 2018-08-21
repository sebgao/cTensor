import numpy as np
from ctensor import Tensor
from ctensor.optim import Adam
from ctensor.functional import conv2d, relu


D = Tensor.randn((1, 1, 50, 50))
T = Tensor.randn((1, 1, 1, 1))
W1 = Tensor.randn((1, 32, 3, 3))/(4.5)
W2 = Tensor.randn((32, 1, 3, 3))/(4.5*32)
B1 = Tensor.zeros((1, 32, 1, 1))
B2 = Tensor.zeros((1, 1, 1, 1))
adam = Adam([W1, W2])
for _ in range(1000):
    I = conv2d(D, W1)#, padding=(1, 1))
    I = relu(I)
    I = conv2d(I, W2)#, padding=(1, 1))
    #I = (I)
    loss = ((I-T)**2)
    adam.zero_grad()
    loss.backward()
    adam.step(1e-5)
    #print(data.grad)
    #print(weight.grad)

print(loss.mean())

# W1 = Tensor.randn((1, 100, 3, 3))/3.5
# W2 = Tensor.randn((100, 1, 3, 3))/(3.5*100)
# B1 = Tensor.zeros((1, 100, 3, 3))
# adam = Adam([W1, W2, B1])
# for _ in range(1000):
#     #print(data.data.std())
#     I1 = conv2d(data, W1)+B1
#     #print(I.data.std(), I.data.mean())
#     I2 = leaky_relu(I1)
#     #print(I.data.std(), I.data.mean())
#     I3 = conv2d(I2, W2)
#     #print(I.data.std(), I.data.mean())
#     K = (I3-1)**2
#     adam.zero_grad()
#     K.backward()
#     adam.step(1e-3)

# print(K)
# print(I1.data.std(), I1.grad.std())
# print(I2.data.std(), I2.grad.std())
# print(I3.data.std(), I3.grad.std())
