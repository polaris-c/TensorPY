import TensorPY as tp
import numpy as np
import matplotlib.pyplot as plt


red_points = np.random.randn(5, 2)  - 2*np.ones((5, 2))
blue_points = np.random.randn(5, 2) + 2*np.ones((5, 2))

tp.Graph().as_default()
a = tp.Variable([[2,1], [-1,-2]])
b = tp.Variable([1, 1])
c = tp.Placeholder()

y = tp.matmul(a, b)
z = tp.add(y, c)
print(z)

session = tp.Session()
output = session.run(z, {c:[3, 3]})
print(output)

print(len(blue_points))
print(len(red_points))
c = [[1,0]]*len(blue_points) + [[0, 1]]*len(red_points)
print(c)

X = np.concatenate((blue_points, red_points))
print(X)