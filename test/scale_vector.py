import tensorlib
import numpy as np

# Test with a sample vector
x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
alpha = 2.0
tensorlib.scale_vector(x, alpha)
print("Scaled vector:", x)
