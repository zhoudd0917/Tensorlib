import tensorlib as tl

device = tl.Device.GPU

x = tl.Tensor(
    [[[1.0, 2.0, 3.0], [10.0, 11.0, 12.0]], [[20.0, 21.0, 22.0], [30.0, 31.0, 32.0]]],
    device,
    requires_grad=True,
)

y = tl.Tensor(
    [[[1.0], [-2.0], [5.0]], [[3.0], [2.0], [1.0]]], device, requires_grad=True
)

z = tl.matmul(x, y)
w = tl.log(z)
l = tl.sum(w, axis=1)

print("x: ", x)
print("y: ", y)
print("z: ", z)
print("w: ", w)
print("l: ", l)

init_grad_l = tl.randn(l.shape, 0.0, 1.0, device, requires_grad=False)

l.backward(init_grad_l)

print("l grad: ", l.grad)
print("w grad: ", w.grad)
print("z grad: ", z.grad)
print("y grad: ", y.grad)
print("x grad: ", x.grad)
