# KNN_CUDA

A modification version of KNN_CUDA

+ original version: [kNN-CUDA](https://github.com/unlimblue/KNN_CUDA)


#### Modifications 
We add two new lines ```line4: #include <c10/cuda/CUDAGuard.h>``` and ```line42: at::cuda::CUDAGuard device_guard(ref.device());``` in ```csrc\cuda\knn.cpp``` to support GPU specify.

*** Note that the original version does not support GPU specify. Once you specify a gpu other than "cuda:0", you will get wrong results ***


#### Install


+ from source (only support)

```bash
git clone https://github.com/neuralchen/KNN_CUDA.git
cd KNN_CUDA
make && make install
```

+ for windows

You should use branch `windows`:

```bash
git clone --branch windows https://github.com/neuralchen/KNN_CUDA.git
cd C:\\PATH_TO_KNN_CUDA
make
make install
```

#### Usage

```python
import torch

# Make sure your CUDA is available.
assert torch.cuda.is_available()

from knn_cuda import KNN
"""
if transpose_mode is True, 
    ref   is Tensor [bs x nr x dim]
    query is Tensor [bs x nq x dim]
    
    return 
        dist is Tensor [bs x nq x k]
        indx is Tensor [bs x nq x k]
else
    ref   is Tensor [bs x dim x nr]
    query is Tensor [bs x dim x nq]
    
    return 
        dist is Tensor [bs x k x nq]
        indx is Tensor [bs x k x nq]
"""

knn = KNN(k=10, transpose_mode=True)

ref = torch.rand(32, 1000, 5).cuda("cuda:1")
query = torch.rand(32, 50, 5).cuda("cuda:1")

dist, indx = knn(ref, query)  # 32 x 50 x 10
```
