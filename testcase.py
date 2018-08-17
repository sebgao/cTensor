import numpy as np
from ctensor import Tensor, conv2d, Adam, relu, SGD, leaky_relu, _MaxPooling, _Conv2d, sigmoid


D = np.array(
    [
    [
        [0.1, 0.2, 0.3],
        [0.1, 0.2, 0.2],
        [0.1, 0.1, 1],
    ]
    ]
    ).reshape(1, 1, 3, 3)

T = np.array(
    [
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ]
).reshape(1, 1, 3, 3)
#print(data, weight)
D = Tensor.randn((1, 1, 3, 3))
T = Tensor(T)
W1 = Tensor.randn((1, 100, 3, 3))/(4.5)
W2 = Tensor.randn((100, 1, 3, 3))/(4.5*100)
B1 = Tensor.zeros((1, 100, 3, 3))
B2 = Tensor.zeros((1, 1, 3, 3))
adam = Adam([W1, W2, B1, B2])
for _ in range(1000):
    I = conv2d(D, W1, padding=(1, 1))+B1
    I = leaky_relu(I)
    I = conv2d(I, W2, padding=(1, 1))+B2
    I = sigmoid(I)
    adam.zero_grad()
    ((I-T)**2).backward()
    adam.step(1e-2)
    #print(data.grad)
    #print(weight.grad)
    print(I)

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
