import numpy as np
from ctensor import Tensor
import ctensor.functional as F
import ctensor.nn as nn
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

# class Model(nn.Module):
#     def __init__(self):
#         self.linear_1 = nn.Linear(5, 100)
#         self.linear_2 = nn.Linear(100, 2)

#     def forward(self, x):
#         return F.sigmoid(self.linear_2(F.leaky_relu(self.linear_1(x))))

class Model(nn.Module):
    def __init__(self):
        self.main = nn.Sequential(
            nn.Linear(5, 1000),
            nn.ReLU(),
            nn.Linear(1000, 2),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.main(x)

model = Model()
adam = Adam(model.parameters())

for _ in range(3000):
    indices = np.random.randint(0, 1000, size=128)
    X_ = X[indices]
    Y_ = Y[indices]
    pred_y = model(X_)
    loss = F.binary_cross_entropy(pred_y, Y_)
    adam.zero_grad()
    loss.backward()
    adam.step(1e-3)
    if _ % 1000 == 0:
        print(loss)

pred_y = model(X)
print(((pred_y.data > 0.5) == Y.data).mean())
