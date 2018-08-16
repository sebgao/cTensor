import numpy as np

def T(x):
    '''Transpose a 3-d matrix, [batch x N x M].'''

    return x.transpose(0, 2, 1)

def makeTensorLike(another, sel):
    '''Make the operand a tensor.'''

    assert isinstance(another, int) or isinstance(another, float) or isinstance(another, Tensor), 'Cannot convert to tensor'

    if isinstance(another, int) or isinstance(another, float):
        return Tensor(np.zeros_like(sel.data)+another)
    return another

def attach_grad(u, grad):
    '''
    During backpropagation, attach computed grad into precedents.
    If forward process includes broadcasting, unbroadcast grad.
    '''

    if u.grad.shape == grad.shape:
        u.grad += grad
        return
    
    # unbroadcasting
    for dim, chn in enumerate(u.grad):
        if u.grad.shape[dim] != grad.shape[dim]:
            assert u.grad.shape[dim] == 1, "Backward unbroadcasting errors"
            grad = grad.mean(axis = dim, keepdims = True)
            
    u.grad += grad
    

class Tensor:
    def __init__(self, ndarray, precedents=None, operator=None, requires_grad=True):
        self.data = ndarray.astype(np.float)
        self.grad = np.zeros_like(self.data)
        self.precedents = precedents
        self.operator = operator
        self.requires_grad = requires_grad
        if precedents:
            self.leaf = False
        else:
            self.leaf = True
            
    def backward(self, internal=False):
        if not internal:
            self.grad = np.ones_like(self.data)
            
        if self.leaf:
            return
        
        if isinstance(self.operator, Operator):
            self.operator.backward(self, self.precedents)

        elif self.operator == 'neg':
            u, = self.precedents
            u.grad += -self.grad
        
        elif self.operator == 'abs':
            u, = self.precedents
            u.grad += self.grad * np.sign(u.data)
        
        elif self.operator == 'exp':
            u, = self.precedents
            u.grad += self.grad * self.data
            
        elif self.operator == 'log':
            u, = self.precedents
            u.grad += self.grad * (1.0/u.data)
        
        elif self.operator == 'sum':
            u, = self.precedents
            u.grad += self.grad
            
            
        elif self.operator == 'mean':
            u, = self.precedents
            elements = 1
            for s in u.grad.shape:
                elements *= s
            u.grad += self.grad / elements
            
        elif self.operator == 'slice':
            u, slic = self.precedents
            u.grad[slic] += self.grad
            
        elif self.operator == '+':
            u, v = self.precedents
            attach_grad(u, self.grad)
            attach_grad(v, self.grad)
            
        elif self.operator == '-':
            u, v = self.precedents
            attach_grad(u, self.grad)
            attach_grad(v, -self.grad)
        
        elif self.operator == '/':
            u, v = self.precedents
            attach_grad(u, -self.grad * v.data / (u.data**2))
            
        elif self.operator == '*':
            u, v = self.precedents
            attach_grad(u, self.grad * v.data)
            attach_grad(v, self.grad * u.data)
            
        elif self.operator == '**':
            u, v = self.precedents
            attach_grad(u, self.grad * u.data**(v.data-1) * v.data)
            
        elif self.operator == 'greater':
            u, v = self.precedents
            attach_grad(u, self.grad * (u.data>v.data))
            
        elif self.operator == '@':
            u, v = self.precedents
            if len(self.data.shape) == 3:
                attach_grad(u, self.grad @ T(v.data))
                attach_grad(v, T(u.data) @ self.grad)
            else:
                attach_grad(u, self.grad @ v.data.T)
                attach_grad(v, u.data.T @ self.grad)
                
            
        for p in self.precedents:
            if isinstance(p, Tensor) and p.requires_grad:
                p.backward(internal=True)
    
    def __neg__(self):
        return Tensor(-self.data, precedents=[self], operator='neg')
    
    def __pos__(self):
        return self
    
    def abs(self):
        return Tensor(np.abs(self.data), precedents=[self], operator='abs')
    
    def sum(self, dim=None):
        return Tensor(np.sum(self.data, dim), precedents=[self], operator='sum')
    
    def mean(self, dim=None):
        return Tensor(np.mean(self.data, dim), precedents=[self], operator='mean')
    
    def exp(self):
        return Tensor(np.exp(self.data), precedents=[self], operator='exp')
    
    def log(self):
        return Tensor(np.log(self.data), precedents=[self], operator='log')
    
    def __add__(self, another):
        another = makeTensorLike(another, self)
        return Tensor(self.data + another.data, precedents=[self, another], operator='+')
    
    def __radd__(self, another):
        return Tensor.__add__(self, another)
    
    def __sub__(self, another):
        another = makeTensorLike(another, self)
        return Tensor(self.data - another.data, precedents=[self, another], operator='-')
    
    def greater(self, another):
        another = makeTensorLike(another, self)
        return Tensor(self.data > another.data, precedents=[self, another], operator='greater', requires_grad=False)
    
    def __rsub__(self, another):
        another = makeTensorLike(another, self)
        return Tensor.__sub__(another, self)
    
    def __pow__(self, another):
        another = makeTensorLike(another, self)
        another.requires_grad = False
        return Tensor(self.data ** another.data, precedents=[self, another], operator='**')
      
    def __truediv__(self, another):
        assert isinstance(another, int) or isinstance(another, float),         'Right divide only supports int or float. Please use *'
        another = makeTensorLike(another, self)
        another.data = 1.0/another.data
        return Tensor.__mul__(self, another)
    
    def __rtruediv__(self, another):
        another = makeTensorLike(another, self)
        return Tensor(another.data / self.data, precedents=[self, another], operator='/')
    
    def __mul__(self, another):
        another = makeTensorLike(another, self)
        return Tensor(self.data * another.data, precedents=[self, another], operator='*')
    
    def __rmul__(self, another):
        return Tensor.__mul__(self, another)
    
    def __matmul__(self, another):
        return Tensor(self.data @ another.data, precedents=[self, another], operator='@')
    
    def __getitem__(self, slic):
        return Tensor(self.data[slic], precedents=[self, slic], operator='slice')
    
    def view(self, *shape):
        return View(shape)(self)
    
    def __repr__(self):
        return "Tensor({})".format(self.data)

    @staticmethod
    def zeros(args):
        return Tensor(np.zeros(args))
    
    @staticmethod
    def randn(args):
        return Tensor(np.random.randn(*args))
    
class Optimizer:
    def __init__(self, parameters):
        assert parameters, 'Your parameters?'
        self.parameters = parameters
        
    def zero_grad(self):
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)
    
    def step(self, lr=0.01):
        assert False, 'Optimizer class is virtual'

class SGD(Optimizer):
    def step(self, lr=0.01):
        for p in self.parameters:
            p.data -= lr*p.grad
            
class Adam(Optimizer):
    def __init__(self, parameters, beta1 = 0.9, beta2 = 0.99, eta = 1e-6):
        super(Adam, self).__init__(parameters)
        self.m = []
        self.v = []
        for p in self.parameters:
            self.m.append(np.zeros_like(p.grad))
            self.v.append(np.zeros_like(p.grad))
        
        self.beta1 = beta1
        self.beta2 = beta2
        self.eta = eta
        
    def step(self, lr=0.01):
        '''
        Adam optimizer stepping.
        We ignore the unbiasing process as it hurts efficiency.
        '''

        beta1 = self.beta1
        beta2 = self.beta2
        for idx, p in enumerate(self.parameters):
            m = self.m[idx]
            v = self.v[idx]
            grad = p.grad
            m = beta1*m + (1-beta1)*grad
            v = beta2*v + (1-beta2)*(grad**2)
            # ignoring unbiasing
            p.data -= lr*m/(np.sqrt(v)+self.eta)

class Operator:
    def forward(self, *args):
        return args[0]
    
    def __call__(self, *args):
        fwd = self.forward(*args)
        if fwd.precedents:
            return Tensor(fwd.data, precedents=[fwd], operator=self) # Operator in Tensor
        else:
            return Tensor(fwd.data, precedents=args, operator=self) # Operation in Numpy
    
    def backward(self, x, precedents):
        p, = precedents
        p.grad = x.grad

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

class _Conv2d(Operator):
    def forward(self, t, weight):
        t = t.data
        w = weight.data
        B, C, iH, iW = t.shape
        iC, oC, kH, kW = w.shape
        assert C == iC, 'Conv2d channels in not equal.'
        return Tensor(batch_conv2d(t, w))
    
    def backward(self, x, precedents):
        t, weight = precedents
        t.grad += batch_conv2d_im_backward(x.grad, weight.data)
        weight.grad += batch_conv2d_weight_backward(x.grad, t.data) 

class View(Operator):
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.origin_shape = x.data.shape
        return Tensor(x.data.reshape(*self.shape))

    def backward(self, x, precedents):
        u, = precedents
        u.grad += x.grad.reshape(self.origin_shape)

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
    return -((1. + 1e-6 -label)*((1. + 1e-6 -pred).log())+(label)*(pred.log())).mean()

def conv2d(x, weight):
    return _Conv2d()(x, weight)

def batch_conv2d(x, kernel):
    x = im2bchwkl(x, kernel.shape[-2:])
    return np.einsum('...chwkl,cvkl->...vhw', x, kernel)


def batch_conv2d_weight_backward(kernel, input):
    '''kernel is result tensor grad, input is original tensor'''
    x = im2bchwkl(input, kernel.shape[-2:])
    return np.einsum('bchwkl,bvkl->bcvhwkl', x, kernel).mean(axis=(6, 5, 0))


def batch_conv2d_im_backward(x, kernel):
    '''input is result tensor grad, kernel is weight tensor'''
    ksize = kernel.shape
    x = im2bchwkl(x, ksize[-2:], padding=(ksize[2]-1, ksize[3]-1))
    return np.einsum('bchwkl,vckl->bvhw', x, kernel)

def im2bchwkl(input, ksize, stride=(1, 1), padding=(0, 0)):
    if padding != (0, 0):
        input = make_padding(input, (padding[0], padding[1]))

    isize = input.shape
    istrides = input.strides

    H = (isize[2]-ksize[0])//stride[0]+1
    W = (isize[3]-ksize[1])//stride[1]+1

    istrides = list(istrides+istrides[-2:])
    istrides[2] *= stride[0]
    istrides[3] *= stride[1]
    return np.lib.stride_tricks.as_strided(input,
        (isize[0], isize[1], H, W, ksize[0], ksize[1]),
        istrides,
        writeable=False,
    )

def make_padding(input, padding):
    b, c, h, w = input.shape
    result = np.zeros((b, c, h+2*padding[0], w+2*padding[1]), dtype=np.float)
    result[:, :, padding[0]:-padding[0], padding[1]:-padding[1]] = input
    return result
