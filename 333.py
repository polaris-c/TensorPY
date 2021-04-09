import numpy as np
import matplotlib.pyplot as plt

import TensorPY as tp

red_points = np.random.randn(5, 2)  - 2*np.ones((5, 2))

blue_points = np.random.randn(5, 2) + 2*np.ones((5, 2))

# print(np.random.randn(50, 2))
# print(2*np.ones((50, 2)))
# print(red_points.shape)
# print(blue_points)

plt.scatter(red_points[:,0], red_points[:,1], color='red')
plt.scatter(blue_points[:,0], blue_points[:,1], color='blue')
plt.show()

tp.Graph().as_default()

X = tp.Placeholder()
W = tp.Variable([[1,-1], [1, -1]])

b = tp.Variable([0,0])
o = tp.add(tp.matmul(X, W), b)
print(tp.add(tp.matmul(X, W), b))

p = tp.sigmod(tp.add(tp.matmul(X, W), b))

session = tp.Session()

output_o = session.run(o, { X:np.concatenate((blue_points, red_points)) })
output_p = session.run(p, { X:np.concatenate((blue_points, red_points)) })

print(output_o)
print(output_p)