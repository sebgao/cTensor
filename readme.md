# cTensor
The cTensor (crafted tensor) is a super light-weight deep learning library (perhaps we cannot even call it a libray). It's based on numpy and furthermore its only one dependency is actually numpy. Features include dynamic graph, autograd and user-defined operations in numpy. The line number of core code is within ~~300~~ 400, making it friendly for study and teaching purpose. It mimics PyTorch framework defacto.

Stars are welcomed! : )

### 2018.8.14
Of course, cTensor has not yet support convolutional operations.
### 2018.8.16
We have supported batch __conv2d__ operation in pytorch fashion (limited in stride 1, no padding) !!