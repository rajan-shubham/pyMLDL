# -*- coding: utf-8 -*-
"""autograd.ipynb

Original file is located at
    https://colab.research.google.com/drive/1mdt_47dHALF8lttJbzEc087-q0ud6w3E
"""

# pyTorch autograd -> pytorch automatic differentiation tool to calclulate derivative
# y = x^2 (write a program so that at any given x find the derivative of y w.r.t. x)
# dy/dx = 2x is differentiation of y
# so at any x say:3 the derivative (slope of y) of y is 2*x i.e. 2*3=>6
def dy_dx(x):
  return 2*x

dy_dx(5)

# write a program which find for any x -> dx/dx for
# y = x^2
# z = sin(y)
# dz/dx = dz/dy * dy/dx (chain rule of diff.)
# dz/dy = cos(y) and dy/dx = 2x so dz/dx = cos(y) * 2x => 2x * cos(x^2)
import math

def dz_dx(x):
  return 2*x*math.cos(x**2)

dz_dx(3)

# y=x^2 z=sin(y) u=e^z
# du/dz = du/dz * dz/dy * dy/dx
# so use autograd (automatic diff./gradient)

import torch

x = torch.tensor(3.0, requires_grad=True)
print(x)

y = x**2
print(y)

y.backward()

x.grad

x = torch.tensor(3.0, requires_grad=True)
print(x)

y = x**2
z = torch.sin(y)

print(y)
print(z)

z.backward()

x.grad

x = torch.tensor(6.7)
y = torch.tensor(0.0)

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

print(w)
print(b)

z = w*x + b
print(z)

y_pred = torch.sigmoid(z)
print(y_pred)

loss = torch.binary_cross_entropy_with_logits(y_pred, y)
print(loss)

loss.backward()

print(w.grad)
print(b.grad)

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
x

y = (x**2).mean()
y

y.backward()

x.grad

# CLEARING GRAD
# if you are running multiple times backward pass then your gradients are accumulating
x.grad.zero_()

# DISABLE GRADIENT TRACKING
# Remove gradient tracking when your model is fully trained, and currently doing prediction
# after 2nd forward pass the gradient is not clearing automatically (it accumulate)
# added with previous gradient
x = torch.tensor(3.0, requires_grad=True)
x

y = x ** 2
y

y.backward()

x.grad

x.grad.zero_() # doing inplace(as it is underscore) changes to 0

# if training is end, now you need only forward pass, you don't need to call backward functions
# in that case you need to disable gradient tracking as it occupy unnecessary memory (large NN -> large memory consume)

# option 1 -> requires_grad_(False)
# option 2 -> torch.detach()
# option 3 -> torch.no_grad()

x.requires_grad_(False)
x

# in detach you have to make a whole new tensor
# x and z are same tensor in terms of data, but x is in couputational graph but z is not
# so you can do y.backward() on x but not on z
x = torch.tensor(3.0, requires_grad=True)
x
z = x.detach()
z

y = x ** 2
y

y1 = z ** 2
y1

# 3. with torch.no_grad()
x = torch.tensor(2.0, requires_grad= True)
x

with torch.no_grad():
  y = x ** 2

y