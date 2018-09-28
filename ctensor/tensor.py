import numpy as np

def T(x):
    '''Transpose a 3-d matrix, [batch x N x M].'''

    return x.transpose(0, 2, 1)


def make_tensor_like(another, sel):
    '''Make the operand a tensor.'''

    assert isinstance(another, int) or isinstance(another, float) or isinstance(
        another, Tensor), 'Cannot convert to tensor'

    if isinstance(another, int) or isinstance(another, float):
        s = (1,)*len(sel.data.shape)
        return Tensor(np.zeros(s)+another, requires_grad=False)
    return another


def attach_grad(u, grad):
    '''
    During backpropagation, attach computed grad into precedents.
    If forward process includes broadcasting, unbroadcast grad.
    '''
    if not u.requires_grad:
        return

    if u.grad.shape == grad.shape:
        u.grad += grad
        return

    # unbroadcasting
    for dim, chn in enumerate(u.grad.shape):
        if chn != grad.shape[dim]:
            assert chn == 1, "Backward unbroadcasting errors"
            grad = grad.mean(axis=dim, keepdims=True)

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
            attach_grad(u, self.grad * (u.data > v.data))

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
        another = make_tensor_like(another, self)
        return Tensor(self.data + another.data, precedents=[self, another], operator='+')

    def __radd__(self, another):
        return Tensor.__add__(self, another)

    def __sub__(self, another):
        another = make_tensor_like(another, self)
        return Tensor(self.data - another.data, precedents=[self, another], operator='-')

    def greater(self, another):
        another = make_tensor_like(another, self)
        return Tensor(self.data > another.data, precedents=[self, another], operator='greater', requires_grad=False)

    def __rsub__(self, another):
        another = make_tensor_like(another, self)
        return Tensor.__sub__(another, self)

    def __pow__(self, another):
        another = make_tensor_like(another, self)
        another.requires_grad = False
        return Tensor(self.data ** another.data, precedents=[self, another], operator='**')

    def __truediv__(self, another):
        assert isinstance(another, int) or isinstance(
            another, float),         'Right divide only supports int or float. Please use *'
        another = make_tensor_like(another, self)
        another.data = 1.0/another.data
        return Tensor.__mul__(self, another)

    def __rtruediv__(self, another):
        another = make_tensor_like(another, self)
        return Tensor(another.data / self.data, precedents=[self, another], operator='/')

    def __mul__(self, another):
        another = make_tensor_like(another, self)
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


class Operator:
    def forward(self, *args):
        return args[0]

    def __call__(self, *args):
        fwd = self.forward(*args)
        if fwd.precedents:
            # Operator in Tensor
            return Tensor(fwd.data, precedents=[fwd], operator=self)
        else:
            # Operation in NumPy
            return Tensor(fwd.data, precedents=args, operator=self)

    def backward(self, x, precedents):
        p, = precedents
        p.grad = x.grad


class View(Operator):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.origin_shape = x.data.shape
        return Tensor(x.data.reshape(*self.shape))

    def backward(self, x, precedents):
        u, = precedents
        u.grad += x.grad.reshape(self.origin_shape)
