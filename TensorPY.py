import numpy as np
try:
    import Queue
except ImportError:
    import queue as Queue
# import queue as 

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
class RegisterGradient(object):
    '''
     给operation注册梯度计算函数
    '''
    def __init__(self, op_type):
        print('-> RegisterGradient op_type: ', op_type, type(op_type))
        # 操作节点类内存地址
        self._op_type = eval(op_type)
    def __call__(self, f):
        print('-> RegisterGradient self._op_type: ', self._op_type)
        _gradient_registry[self._op_type] = f
        return f

@RegisterGradient('negative')
def _negative_gradient(op, grad):
    '''
    '''
    print('-> _negative_gradient grad:', grad)
    return -grad

@RegisterGradient('reduce_sum')
def _reduce_sum_gradient(op, grad):
    '''
    '''
    print('-> _reduce_sum_gradient grad:', grad)
    return -grad

@RegisterGradient('multiply')
def _multiply_gradient(op, grad):
    '''
    '''
    print('-> _multiply_gradient grad:', grad)
    return -grad

# @RegisterGradient('reduce_sum')
# def _reduce_sum_gradient(op, grad):
#     '''
#     '''
#     print('-> _reduce_sum_gradient grad:', grad)
#     return -grad

print('-> _gradient_registry_: ')
# print(str(_gradient_registry))
_gradient_registry_ = _gradient_registry.items()
for k, v in _gradient_registry_:
    print('\t %s -- %s' % (k, v))

def computer_gradients(loss):
    grad_table = {}
    grad_table[loss] = 1
    visited = set()
    queue = Queue.Queue()
    visited.add(loss)
    queue.put(loss)

    while not queue.empty():
        node = queue.get()
        if node != loss:
            grad_table[node] = 0
            for consumer in node.consumers:
                lossgrad_wrt_consumer_output = grad_table[consumer]
                consumer_op_type = consumer.__class__
                bprop = _gradient_registry[consumer_op_type]
                lossgrad_wrt_consumer_input = bprop(consumer, lossgrad_wrt_consumer_output)

                if len(consumer.input_nodes) == 1:
                    grad_table[node] += lossgrad_wrt_consumer_input
                else:
                    node_index_in_consumer_inputs = consumer.input_node.index(node)
                    lossgrad_wrt_node = lossgrad_wrt_consumer_input[node_index_in_consumer_inputs]
                    grad_table[node] += lossgrad_wrt_node
        if hasattr(node, 'input_nodes'):
            for input_node in node.input_nodes:
                if not input_node in visited:
                    visited.add(input_node)
                    queue.put(input_node)
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
        print('-> run operation:', operation)
        nodes_postorder = traverse_postorder(operation)
        # print('-> nodes_postorder: ', nodes_postorder)
        for node in nodes_postorder:
            print('-> node: ', node)
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