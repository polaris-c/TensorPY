import numpy as np
import matplotlib.pyplot as plt

import TensorPY as tp

red_points = np.random.randn(5, 2)  - 2*np.ones((5, 2))
blue_points = np.random.randn(5, 2) + 2*np.ones((5, 2))

tp.Graph().as_default()

W = tp.Variable(np.random.randn(2, 2))
b = tp.Variable(np.random.randn(2))

X = tp.Placeholder()
c = tp.Placeholder()

p = tp.softmax(tp.add(tp.matmul(X, W), b))

feed_dict = {
    X: np.concatenate((blue_points, red_points)),
    c: [[1,0]]*len(blue_points) + [[0, 1]]*len(red_points)
}

print('>> Data is OK')

J = tp.negative(tp.reduce_sum(tp.reduce_sum(tp.multiply(c, tp.log(p)), axis=1)))

minimization_op = tp.GradientDescentOptimizer(learning_rate=0.01).minimize(J)

print('>> J and minimization_op is OK')

print('>> Session ')
session = tp.Session()
for step in range(5):
    J_value = session.run(J, feed_dict)
    if step % 1 == 0:
        print('Step:[%s], Loss:[%s]' % (step, J_value))
session.run(minimization_op, feed_dict)

W_value = session.run(W)
print('Weight matrix:\n', W_value)
b_value = session.run(b)
print('Bias:\n', b_value)

plt.scatter(red_points[:,0], red_points[:,1], color='red')
plt.scatter(blue_points[:,0], blue_points[:,1], color='blue')

x_axis = np.linspace(-4, 4, 50)
y_axis = -W_value[0][0] / W_value[1][0] * x_axis - b_value[0]/W_value[1][0]
plt.show()