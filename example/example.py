import tensorlib as tl
import sys

if len(sys.argv) > 1:
    if sys.argv[1] == "cpu":
        device = tl.Device.CPU
    elif sys.argv[1] == "gpu":
        device = tl.Device.GPU
    else:
        print("Invalid device argument. Usage: python example.py [cpu|gpu]")
        sys.exit(1)

x = tl.Tensor(
    [[[1.0, 2.0, 3.0], [-2.0, 1.0, 2.1]], [[2.0, 3.0, -7.0], [2.0, -1.0, -2.0]]],
    device,
    requires_grad=True,
)

y = tl.Tensor(
    [
        [[1.0, 1.2, 5.2, 3.0], [-2.0, 1.0, 2.1, 1.0], [1.0, 2.0, 3.0, 1.0]],
        [[2.0, 3.0, 7.0, 1.0], [2.0, -1.0, -2.0, 1.0], [1.0, 2.0, 3.0, 1.0]],
    ],
    device,
    requires_grad=True,
)

print(x.shape)
print(y.shape)
print("x: ", x)
print("y: ", y)

z = tl.matmul(x, y)

print("z: ", z)
w = tl.softmax(z, axis=1)
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
