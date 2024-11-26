import tensorlib as tl
from tensorlib import Device
import numpy as np

device = Device.GPU  # Change to Device.CPU if needed

x = tl.Tensor(
    np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).reshape(
        2, 2, 3
    ),
    requires_grad=True,
    device=device,
)

y = tl.Tensor(
    np.array(
        [
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
            16.0,
            17.0,
            18.0,
            19.0,
            20.0,
            21.0,
            22.0,
            23.0,
            24.0,
            -2.0,
        ]
    ).reshape(2, 3, 4),
    requires_grad=True,
    device=device,
)

z = tl.matmul(x, y)
w = z + z
l = tl.transpose(w)

# Print tensors
print("x: ", x)
print("y: ", y)
print("z: ", z)
print("w: ", w)
print("l: ", l)

# Initialize the gradient for w
init_grad_w = tl.Tensor(
    np.array(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
            16.0,
        ]
    ).reshape(2, 4, 2),
    device=device,
)

# Perform backward pass for gradient calculation
l.backward(init_grad_w)

# Print gradients
print("l grad: ", l.grad)
print("z grad: ", z.grad)
print("w grad: ", w.grad)
print("y grad: ", y.grad)
print("x grad: ", x.grad)
