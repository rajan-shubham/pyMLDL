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
x = torch.tensor(3.0, requires_grad=True)
x

y = x ** 2
y

y.backward()

x.grad

# option 1 -> requires_grad_(False)
# option 2 -> torch.detach()
# option 3 -> torch.no_grad()

x.requires_grad_(False)
x

x = torch.tensor(3.0, requires_grad=True)
x
z = x.detach()
z

y = x ** 2
y

y1 = z ** 2
y1

x = torch.tensor(2.0, requires_grad= True)
x

with torch.no_grad():
  y = x ** 2

y
