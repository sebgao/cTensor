# cTensor
The cTensor (crafted tensor) is a super light-weight deep learning library (perhaps we cannot even call it a libray). It's based on numpy and furthermore its only one dependency is actually numpy. Features include dynamic graph, autograd and user-defined operations in numpy. The line number of core code is within ~~300~~ 400, making it friendly for study and teaching purpose. It mimics PyTorch framework defacto.

Stars are welcomed! : )

## File Structure

### ctensor/tensor.py

The file contains the definition of `Tensor` and its many basic operations. The definition of `Operator` enables user-defined operators on `Tensor` and therefore `forward` and `backward` in `numpy.ndarray` or `Tensor` (see example `View` class).

### ctensor/operator.py

The file includes some pre-defined `Operator`s on `Tensor`, e.g., `ReLU` and `Conv2d`. Note that misc things about `Conv2d` are exposed here as I am lazy to separate details and abstracts about it.

### ctensor/functional.py

An `Operator` subclass could be applied to `Tensor` only when an instance of the class are initiated. The file consists of functions that make an instance of the `Operator` and pass through it.

### ctensor/optim.py

`Optimizer`s here.

### ctensor/nn.py

High level abstracts of `Operator`s and their parameters. They are totally in PyTorch fashion.

### testcase/*.py

Some testcases. Maybe too hard to read although I've written up some comments.


## Update

### 2018.11.2
Readme in more detail.

### 2018.9.28
Add `nn.Conv2d`.

### 2018.8.18
Speed 100x up `conv2d` by using `np.tensordot`, which operates automatically in multicores.

### 2018.8.17
~~Now support __max_pooling__ operation with any stride, kernel size and padding.~~

Fix buges related to `conv2d` backpropagation.
Now support `conv2d` with any stride and padding.
### 2018.8.16
We have supported batch `conv2d` operation in pytorch fashion (limited in stride 1, no padding) !!
### 2018.8.14
~~Of course, cTensor has not yet support convolutional operations.~~
