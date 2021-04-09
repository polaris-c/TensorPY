import TensorPY as tp

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