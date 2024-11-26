import tensorlib
from tensorlib import Device
import numpy as np

x = tensorlib.Tensor([1, 2, 3, 4], shape=(2, 2), requires_grad=True, device=Device.GPU)
y = tensorlib.Tensor([5, 6, 7, 8], shape=(2, 2), requires_grad=True, device=Device.GPU)

z = x + y
w = z + x
w.backward(y)

print("x: ", x)
print("y: ", y)
print("z: ", z)
print("w: ", w)

print("w.grad: ", w.grad)
print("z.grad: ", z.grad)
print("y.grad: ", y.grad)
print("x.grad: ", x.grad)
