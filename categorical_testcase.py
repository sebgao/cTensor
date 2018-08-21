import numpy as np
from ctensor import Tensor
from ctensor.functional import leaky_relu, sigmoid, binary_cross_entropy
from ctensor.optim import Adam

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
M1 = Tensor(np.random.randn(5, 100)/500)
B1 = Tensor(np.zeros((1, 100)))

M2 = Tensor(np.random.randn(100, 1)/100)
B2 = Tensor(np.zeros((1, 1)))

# Optimizer
adam = Adam([M1, B1, M2, B2])

# Compute Graph Definition, note that the graph is actually dynamic

def compute(X):
    X1 = leaky_relu((X @ M1) + B1, 0.01)
    X2 = sigmoid((X1 @ M2) + B2)
    return X2

for _ in range(10000):
    indices = np.random.randint(0, 1000, size=128)
    X_ = X[indices]
    Y_ = Y[indices]
    pred_y = compute(X_)
    loss = binary_cross_entropy(pred_y, Y_)
    adam.zero_grad()
    loss.backward()
    adam.step(1e-3)
    if _ % 1000 == 0:
        print(loss)

# If succeeded, the score would be relatively high
pred_y = compute(X)
print(((pred_y.data > 0.5) == Y.data).mean())
