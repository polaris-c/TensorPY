import numpy as np

# import OperationClass

class Graph(object):
    '''
    '''
    def __init__(self):
        self.operations = []
        self.placeholders = []
        self.variables = []
    def as_default(self):
        global _default_graph
        _default_graph = self

class Placeholder(object):
    '''
    '''
    def __init__(self):
        self.consumers = []
        _default_graph.placeholders.append(self)

class Variable(object):
    '''
    '''
    def __init__(self, input_value=None):
        self.value = input_value
        self.consumers = []
        _default_graph.variables.append(self)

class Operation(object):
    '''
    '''
    def __init__(self, input_nodes=[]):
        self.input_nodes = input_nodes
        self.consumers = []
        for input_node in input_nodes:
            input_node.consumers.append(self)
            _default_graph.operations.append(self)
    def computer(self):
        pass

# 
class matmul(Operation):
    '''
    '''
    def __init__(self, x, y):
        super().__init__([x, y])
    def computer(self, x_value, y_value):
        return x_value.dot(y_value)

class add(Operation):
    '''
    '''
    def __init__(self, x, y):
        super().__init__([x, y])
    def computer(self, x_value, y_value):
        return x_value + y_value

class sigmod(Operation):
    '''
    '''
    def __init__(self, a):
        super().__init__([a])
    def computer(self, a_value):
        return 1 / (1 + np.exp(-a_value))

class softmax(Operation):
    '''
    '''
    def __init__(self, a):
        super().__init__([a])
    def computer(self, a_value):
        return np.exp(a_value) / np.sum(np.exp(a_value), axis=1)[:, None]

class negative(Operation):
    '''
    '''
    def __init__(self, x):
        super().__init__([x])
    def computer(self, x_value):
        return -x_value

class reduce_sum(Operation):
    '''
    '''
    def __init__(self, A, axis=None):
        super().__init__([A])
        self.axis = axis
    def computer(self, A_value):
        return np.sum(A_value, self.axis)

class multiply(Operation):
    '''
    '''
    def __init__(self, x, y):
        super().__init__([x, y])
    def computer(self, x_value, y_value):
        return x_value * y_value
    
class log(Operation):
    '''
    '''
    def __init__(self, x):
        super().__init__([x])
    def computer(self, x_value):
        return np.log(x_value)

_gradient_registry = {}


def computer_gradients(loss):
    grad_table = {}

    return grad_table


class GradientDescentOptimizer(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    def minimize(self, loss):
        learning_rate = self.learning_rate
        #Operation
        class MinimizationOperation(Operation):
            def computer(self):
                grad_table = computer_gradients(loss)
                for node in grad_table:
                    if isinstance(node, Variable):
                        grad = grad_table[node]
                        node.value -= learning_rate * grad
        return MinimizationOperation()
  
# 
def traverse_postorder(operation):
    '''
    '''
    nodes_postorder = []
    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)
    recurse(operation)
    return nodes_postorder

class Session(object):
    def run(self, operation, feed_dict={}):
        '''
        '''
        nodes_postorder = traverse_postorder(operation)
        for node in nodes_postorder:
            if isinstance(node, Placeholder):
                node.output = feed_dict[ node ]
            elif isinstance(node, Variable):
                node.output = node.value
            else:
                node.inputs = [ input_node.output for input_node in node.input_nodes ]
                node.output = node.computer(*node.inputs)
            if isinstance(node.output, list):
                node.output = np.array(node.output)
        return operation.output