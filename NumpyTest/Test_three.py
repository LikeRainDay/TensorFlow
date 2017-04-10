import numpy as np

a = np.zeros((2, 5), dtype=np.float32)
print(a)
a[0, :] = 1.0
a[1, :] = 2
print(a)

a = np.mean(a, axis=(0, 1))

print(a)
