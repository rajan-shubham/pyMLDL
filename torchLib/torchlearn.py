# -*- coding: utf-8 -*-
"""TorchLearn.ipynb

Original file is located at
    https://colab.research.google.com/drive/1lBD1S91PNo3OGsSIFSdWeH5KC5UXPOG8
"""

import torch
print(torch.__version__)

if torch.cuda.is_available():
    print("CUDA/GPU is available.")
    print("Number of GPUs available:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print("GPU", i, ":", torch.cuda.get_device_name(i))
else:
    print("CUDA is not available.")

# Creating Tensor
# using empty
a = torch.empty(2,3)
# empthy function create a tensor and alocate a memory of 2,3 which contain garbage value

# check type
type(a)
# using zeros make tensor (helps in creating biases of all the neurons, initially it is 0)
torch.zeros(2,3)
torch.ones(2,3)
torch.rand(2,3) # random value b/w 0 and 1 (used in weight initialization techniques)

# manual_seed
torch.manual_seed(1729)
torch.rand(2,3)

torch.manual_seed(1729)
torch.rand(2,3)

# using tensor
print(torch.tensor([[1,2,3],[4,5,6]]))

# arange (start, end, jump)
print("using arange -> ", torch.arange(0,20,2))

# using linspace (equal linearly spaced values)
print("using linspace -> ", torch.linspace(0,10,5))

# using eye (identity matrix 3*3)
print("using eye -> ", torch.eye(3))

# using full ((shape of tensor), fill with value 7)
print("using full -> ", torch.full((2,3), 7))

x = torch.tensor([[1,2,3], [4,5,6], [7, 8, 9]])
print(x.shape)
print(x.size())

# make same new tensor which has same shape
torch.empty_like(x)
torch.zeros_like(x)
torch.ones_like(x)
torch.rand_like(x, dtype=torch.float64)

# find data type
print(x.dtype)
# assign data type
print(torch.tensor([1.0, 3.0, 5.0], dtype=torch.int64))
print(torch.tensor([[1, 3, 5], [2, 4, 6]], dtype=torch.float64))
print(torch.tensor([[1, 3, 5], [2, 4, 6]], dtype=torch.bool))
# using to()
print(x.to(torch.float32))

x = torch.rand(2,3)
print(x)
# SCALAR OPERATION
print(x.sum())
print(x.max())
print(x.min())
print(x.mean())

# addition
x + 3
# substraction
x - 3
# multiplicaton
x * 5
# division
x / 3
# int division
(x * 100)//3
# mod
((x * 100)//3)%2
# power
x ** 2

x = torch.rand(2,3)
y = torch.rand(2,3)

# add 2 tensor element wise
x + y
# subtract
x - y
# multiply
x * y
# division
x / y
# power
x ** y
# mod
x % y

# abs
c = torch.tensor([1, -2, 3, -4, 5])
print(c.abs())
print(torch.abs(c))

# negative
torch.neg(c)

d = torch.tensor([1.9, 2.3, 3.7, 4.4])
# round
torch.round(d)

# ceil (upper value)
torch.ceil(d)
# floor (lower value)
torch.floor(d)
# clamp (ek range me rakh sakte ho sare no. ko)
torch.clamp(d, min=2, max=4)

# reduction operation
e = torch.randint(size=(2,3), low=0, high=10, dtype=torch.float64)
print(e)

# sum
print(torch.sum(e))
# sum along columns
print(torch.sum(e, dim=0))
# sum along rows
print(torch.sum(e, dim=1))

# mean
print(torch.mean(e))
print(torch.mean(e, dim=0))
print(torch.mean(e, dim=1))

# median
print(torch.median(e))

# max and min
print(torch.max(e))
print(torch.min(e))

# prouct
print(torch.prod(e))

# standard deviation
print(torch.std(e))

# variance
print(torch.var(e))

# argmax
print(torch.argmax(e))
print(torch.argmax(e, dim=0))

# argmin
print(torch.argmin(e))
print(torch.argmin(e, dim=0))

# Matrix operations
f = torch.randint(size=(2,3), low=0, high=10, dtype=torch.float64)
g = torch.randint(size=(3,2), low=0, high=10, dtype=torch.float64)
print(f)
print(g)

# matrix multiplication
torch.matmul(f, g)

vector1 = torch.tensor([2, 3])
vector2 = torch.tensor([5, 8])
# dot product
torch.dot(vector1, vector2)

torch.transpose(f, 0, 1)

h = torch.randint(size=(3,3), low=0, high=10, dtype=torch.float64)
i = torch.randint(size=(3,3), low=0, high=10, dtype=torch.float64)
print(h)
print(i)

# determinant
torch.det(h)

# inverse
torch.inverse(h)

# greater than
h > i
# less than
h < i
# equal to
h == i
# not equal to
h != i

j = torch.randint(size=(3,3), low=0, high=10, dtype=torch.float64)
j

# log
print(torch.log(j))
# exp
print(torch.exp(j))
# sqrt
print(torch.sqrt(j))
# sigmoid
print(torch.sigmoid(j))
# softmax
print(torch.softmax(j, dim=0))
# tanh
print(torch.tanh(j))

print(torch.relu(j))

h.add_(i) # inplace addition a <- a + b

h = i.clone()
h

# Tensor Operations on GPU
torch.cuda.is_available()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# creating a new tensor on gpu
torch.rand((2,3), device=device)

# moving an existing tensor to gpu
x = i.to(device)
x
