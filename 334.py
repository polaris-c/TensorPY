import numpy as np
import matplotlib.pyplot as plt

import TensorPY as tp

red_points = np.random.randn(5, 2)  - 2*np.ones((5, 2))

blue_points = np.random.randn(5, 2) + 2*np.ones((5, 2))

plt.scatter(red_points[:,0], red_points[:,1], color='red')
plt.scatter(blue_points[:,0], blue_points[:,1], color='blue')
plt.show()

tp.Graph().as_default()

X = tp.Placeholder()
W = tp.Variable([[1,-1], [1, -1]])

b = tp.Variable([0,0])
# print(tp.add(tp.matmul(X, W), b))

# p = tp.sigmod(tp.add(tp.matmul(X, W), b))
p = tp.softmax(tp.add(tp.matmul(X, W), b))

session = tp.Session()

output_p = session.run(p, { X:np.concatenate((blue_points, red_points)) })

print(output_p)

J = tp.negative(np.reduce_sum(np.reduce_sum(tp.multiply(c, tp.log(p)), axis=1)))