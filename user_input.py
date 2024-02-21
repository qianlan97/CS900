import numpy as np
import random

# seed = 42
# random.seed(seed)
# np.random.seed(seed)

r = np.random.randint(-1000, 1000, size=(1, 23040), dtype=int)

# input image shape is (1, 3, 32, 32), so the shape after flattening is (1, 3072)
# x = np.random.rand(1, 3072)
x = np.random.randint(-1000, 1000, size=(1, 23040), dtype=int)
# assume next layer has 50 neurons
# w = np.random.rand(3072, 50)
w = np.random.randint(-1000, 1000, size=(23040, 1024), dtype=int)

c = x + r
alpha = np.dot(r, w)
xw = np.dot(x, w)

# print(c.shape)
# print(alpha.shape)
# print(xw.shape)

# Save the three arrays to the .npy file
np.save('user_input.npy', (c, alpha, xw, w))
