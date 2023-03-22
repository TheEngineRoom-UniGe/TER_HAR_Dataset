import numpy as np

x = np.arange(18)
print(x)
x_3d = x.reshape((3,2,3))
print(x_3d)
print(x_3d.shape)
new_shape = x_3d.shape[0] * x_3d.shape[2]
print('\nnew shape:', new_shape)
x_2d = x.reshape((2,new_shape))
print(x_2d)
print(x_2d.shape)