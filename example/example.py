import tensorlib as tl
import sys

if len(sys.argv) > 1 and sys.argv[1] == "gpu":
    device = tl.Device.GPU
else:
    device = tl.Device.CPU

x = tl.Tensor(
    [[1.0, 2.0, 3.0, -2.0], [4.0, 5.0, 6.0, 1.0], [7.0, 8.0, 9.0, 1.0]],
    device,
    requires_grad=True,
)

y = tl.Tensor(
    [
        [1.0, 2.0, 3.0, -6.0],
        [4.0, 5.0, 6.0, 1.0],
        [7.0, 8.0, 9.0, 1.0],
    ],
    device,
    requires_grad=True,
)

print(x.shape)
print(y.shape)
print("x: ", x)
print("y: ", y)

z = tl.cross_entropy(x, y)

print("z: ", z)
w = tl.relu(z) / 10.0
print("w: ", w)
l = tl.exp(w)
print("l: ", l)

init_grad_l = tl.ones(l.shape, device, requires_grad=False)

l.backward(init_grad_l)

print("l grad: ", l.grad)
print("w grad: ", w.grad)
print("z grad: ", z.grad)
print("y grad: ", y.grad)
print("x grad: ", x.grad)
