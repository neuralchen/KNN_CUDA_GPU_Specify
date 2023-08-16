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
# """
# if transpose_mode is True, 
#     ref   is Tensor [bs x nr x dim]
#     query is Tensor [bs x nq x dim]
    
#     return 
#         dist is Tensor [bs x nq x k]
#         indx is Tensor [bs x nq x k]
# else
#     ref   is Tensor [bs x dim x nr]
#     query is Tensor [bs x dim x nq]
    
#     return 
#         dist is Tensor [bs x k x nq]
#         indx is Tensor [bs x k x nq]
# """

knn = KNN(k=3, transpose_mode=True).to("cuda:6")

size = 512
xs = torch.linspace(0, size, steps=size)
ys = torch.linspace(0, size, steps=size)
x, y = torch.meshgrid(xs, ys)
x= x.reshape(-1,1)

y= y.reshape(-1,1)
query = torch.cat((x,y),dim=1).unsqueeze(0).to("cuda:6")

ref = torch.randint(0, size, (1, 6000, 2)).float().to("cuda:6")

dist, indx = knn(ref, query)
print(indx)
print(dist)
```
