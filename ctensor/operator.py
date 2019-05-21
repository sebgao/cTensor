import numpy as np
from .tensor import *

class ReLU(Operator):
    def forward(self, x):
        data = x.data
        self.loc = data >= 0
        return Tensor(data*self.loc)

    def backward(self, x, precedents):
        u, = precedents
        u.grad += x.grad*self.loc


class LeakyReLU(Operator):
    def __init__(self, leaky_rate=0.01):
        self.leaky_rate = leaky_rate

    def forward(self, x):
        data = x.data
        loc = data >= 0
        self.effc = loc + self.leaky_rate*(1-loc)
        return Tensor(data*self.effc)

    def backward(self, x, precedents):
        u, = precedents
        u.grad += x.grad*self.effc


class Sigmoid(Operator):
    def forward(self, x):
        data = x.data
        self.result = 1.0/(1.0+np.exp(-data))
        return Tensor(self.result)

    def backward(self, x, precedents):
        u, = precedents
        u.grad += x.grad*(self.result)*(1-self.result)


class Conv2d(Operator):
    def __init__(self, padding=(0, 0), stride=(1, 1)):
        self.padding = padding
        self.stride = stride

    def forward(self, t, weight):
        t = t.data
        w = weight.data
        t = make_padding(t, self.padding)
        B, C, iH, iW = t.shape
        iC, oC, kH, kW = w.shape
        assert C == iC, 'Conv2d channels in not equal.'
        return Tensor(batch_conv2d_f(t, w, self.stride))

    def backward(self, x, precedents):
        t, weight = precedents

        t.grad += unwrap_padding(
            batch_conv2d_im_backward_f(x.grad, weight.data, self.stride),
            self.padding
        )
        weight.grad += batch_conv2d_weight_backward_f(
            x.grad,
            make_padding(t.data, self.padding),
            self.stride
        )

def batch_conv2d_f(x, kernel, stride=(1, 1)):
    x = im2bchwkl(x, kernel.shape[-2:], stride)
    return np.tensordot(x, kernel, [(1, 4, 5), (0, 2, 3)]).transpose(0, 3, 1, 2)


def batch_conv2d_weight_backward_f(kernel, input, stride=(1, 1)):
    '''kernel is result tensor grad, input is original tensor'''
    B, C, H, W = kernel.shape
    x = im2bchwkl(input, kernel.shape[-2:], dilation=stride)
    return np.tensordot(x, kernel, [(0, 4, 5), (0, 2, 3)]).transpose(0, 3, 1, 2)


def batch_conv2d_im_backward_f(x, kernel, stride=(1, 1)):
    '''input is result tensor grad, kernel is weight tensor'''
    ksize = kernel.shape
    x = dilate_input(x, stride)
    x = make_padding(x, ((ksize[2]-1), (ksize[3]-1)))
    return batch_transposed_conv2d_f(x, kernel, invert=True)


def batch_transposed_conv2d_f(x, kernel, invert=False):
    ksize = kernel.shape
    x = transpose_kernel(
        im2bchwkl(x, ksize[-2:])
    )
    i = 1 if invert else 0
    return np.tensordot(x, kernel, [(1, 4, 5), (i, 2, 3)]).transpose(0, 3, 1, 2)


def im2bchwkl(input, ksize, stride=(1, 1), padding=(0, 0), dilation=(1, 1), writeable=False):
    if padding != (0, 0):
        assert not writeable, 'No writable in padding mode.'
        input = make_padding(input, (padding[0], padding[1]))

    isize = input.shape
    istrides = input.strides

    H = (isize[2]-(dilation[0]*(ksize[0]-1)+1))/(stride[0])+1
    W = (isize[3]-(dilation[1]*(ksize[1]-1)+1))/(stride[1])+1
    assert int(H) == H and int(W) == W, 'conv2d not aligned'
    H = int(H)
    W = int(W)
    istrides = list(istrides+istrides[-2:])
    istrides[2] *= stride[0]
    istrides[3] *= stride[1]
    istrides[4] *= dilation[0]
    istrides[5] *= dilation[1]
    return np.lib.stride_tricks.as_strided(input,
                                           (isize[0], isize[1], H,
                                            W, ksize[0], ksize[1]),
                                           istrides,
                                           writeable=writeable,
                                           )


def make_padding(input, padding):
    if padding == (0, 0):
        return input
    b, c, h, w = input.shape
    p, q = padding
    result = np.zeros((b, c, h+2*p, w+2*q), dtype=np.float)
    result[:, :, p:-p, q:-q] = input
    return result


def unwrap_padding(input, padding):
    if padding == (0, 0):
        return input
    p, q = padding
    return input[..., p:-p, q:-q]


def transpose_kernel(kernel):
    return kernel[..., ::-1, ::-1]


def dilate_input(input, stride=(1, 1)):
    if stride == (1, 1):
        return input
    isize = input.shape
    x = np.zeros((isize[0], isize[1], (isize[2]-1) *
                  stride[0]+1, (isize[3]-1)*stride[1]+1))
    x[..., ::stride[0], ::stride[1]] = input
    return x
