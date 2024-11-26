import tensorlib as tl
from tensorlib import Device
import numpy as np

x = tl.Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True, device=Device.CPU)
y = tl.Tensor([[1, 4], [3, 1], [2, 5]], requires_grad=True, device=Device.CPU)

z = tl.transpose(x) + y
w = tl.matmul(x, z)

w.backward(w)

print("x: ", x)
print("y: ", y)
print("z: ", z)
print("w: ", w)

print("w.grad: ", w.grad)
print("z.grad: ", z.grad)
print("y.grad: ", y.grad)
print("x.grad: ", x.grad)
