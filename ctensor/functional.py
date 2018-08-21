from .operator import Sigmoid, ReLU, LeakyReLU, Conv2d

def sigmoid(x):
    return Sigmoid()(x)
    #return  1.0/(1.0+(-x).exp()) # slower implementation


def relu(x):
    return ReLU()(x)


def leaky_relu(x, leaky_rate=0.01):
    return LeakyReLU(leaky_rate=leaky_rate)(x)


def mean_squared_error(pred, label):
    return ((pred-label)**2).mean()


def binary_cross_entropy(pred, label):
    return -((1. + 1e-6 - label)*((1. + 1e-6 - pred).log())+(label)*(pred.log())).mean()


def conv2d(x, weight, padding=(0, 0), stride=(1, 1)):
    return Conv2d(padding, stride)(x, weight)
