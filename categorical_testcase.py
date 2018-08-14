import numpy as np
from ctensor import Tensor, relu, leaky_relu, sigmoid, Adam, binary_cross_entropy

# Data Generation
x = np.random.rand(1000, 5)*10 - 5
x_scale = np.array([1.2, 2, 0.4, 0.7, 0.2]).reshape(1, 5)
x = x*x_scale
y = x[:, 0]**2-2*x[:, 1]+10*x[:, 2]+0.04*x[:, 3]+np.abs(2*x[:, 4])
y = ((y+np.random.rand(1000)) > 17).astype(np.float)

# Tensor Converting
X = Tensor(x)
Y = Tensor(y.reshape(1000, 1))

# Parameters
M1 = Tensor(np.random.randn(5, 100)/100)
B1 = Tensor(np.random.randn(1, 100)/100)

M2 = Tensor(np.random.randn(100, 1)/20)
B2 = Tensor(np.random.randn(1, 1)/100)

# Optimizer
adam = Adam([M1, B1, M2, B2])

# Compute Graph Definition, note that the graph is actually dynamic

def compute(X):
    X1 = relu((X @ M1) + B1)
    X2 = sigmoid((X1 @ M2) + B2)
    return X2

for _ in range(1000):
    pred_y = compute(X)
    loss = binary_cross_entropy(pred_y, Y)
    adam.zero_grad()
    loss.backward()
    adam.step(1e-3)
    if _ % 100 == 0:
        print(loss)

# If succeeded, the score would be relatively high
pred_y = compute(X)
print(((pred_y.data > 0.5) == Y.data).mean())
