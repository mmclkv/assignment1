import numpy as np

class Node(object):
    """Node in a computation graph."""
    def __init__(self):
        """Constructor, new node is indirectly created by Op object __call__ method.

            Instance variables
            ------------------
            self.inputs: the list of input nodes.
            self.op: the associated op object,
                e.g. add_op object if this node is created by adding two other nodes.
            self.const_attr: the add or multiply constant,
                e.g. self.const_attr=5 if this node is created by x+5.
            self.name: node name for debugging purposes.
        """
        self.inputs = []
        self.op = None
        self.const_attr = None
        self.name = ""

    def __add__(self, other):
        """Adding two nodes return a new node."""
        if isinstance(other, Node):
            new_node = add_op(self, other)
        else:
            # Add by a constant stores the constant in the new node's const_attr field.
            # 'other' argument is a constant
            new_node = add_byconst_op(self, other)
        return new_node

    def __sub__(self, other):
        if isinstance(other, Node):
            new_node = sub_op(self, other)
        else:
            new_node = sub_byconst_op(self, other)
        return new_node

    def __rsub__(self, other):
        if isinstance(other, Node):
            new_node = mul_byconst_op(sub_op(self, other), -1)
        else:
            new_node = mul_byconst_op(sub_byconst_op(self, other), -1)
        return new_node

    def __neg__(self):
        return mul_byconst_op(self, -1)

    def __mul__(self, other):
        """TODO: Your code here"""
        if isinstance(other, Node):
            new_node = mul_op(self, other)
        else:
            new_node = mul_byconst_op(self, other)
        return new_node

    def __truediv__(self, other):
        if isinstance(other, Node):
            new_node = div_op(self, other)
        else:
            new_node = div_byconst_op(self, other)
        return new_node

    # Allow left-hand-side add and multiply.
    __radd__ = __add__
    __rmul__ = __mul__


    def __str__(self):
        """Allow print to display node name."""
        return self.name

    __repr__ = __str__


def Variable(name):
    """User defined variables in an expression.
        e.g. x = Variable(name = "x")
    """
    placeholder_node = placeholder_op()
    placeholder_node.name = name
    return placeholder_node


class Op(object):
    """Op represents operations performed on nodes."""
    def __call__(self):
        """Create a new node and associate the op object with the node.

        Returns
        -------
        The new node object.
        """
        new_node = Node()
        new_node.op = self
        return new_node

    def compute(self, node, input_vals):
        """Given values of input nodes, compute the output value.

        Parameters
        ----------
        node: node that performs the compute.
        input_vals: values of input nodes.

        Returns
        -------
        An output value of the node.
        """
        raise NotImplementedError

    def gradient(self, node, output_grad):
        """Given value of output gradient, compute gradient contributions to each input node.

        Parameters
        ----------
        node: node that performs the gradient.
        output_grad: value of output gradient summed from children nodes' contributions

        Returns
        -------
        A list of gradient contributions to each input node respectively.
        """
        raise NotImplementedError


class AddOp(Op):
    """Op to element-wise add two nodes."""
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s+%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        """Given values of two input nodes, return result of element-wise addition."""
        assert len(input_vals) == 2
        return input_vals[0] + input_vals[1]

    def gradient(self, node, output_grad):
        """Given gradient of add node, return gradient contributions to each input."""
        return [output_grad, output_grad]


class AddByConstOp(Op):
    """Op to element-wise add a nodes by a constant."""
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s+%s)" % (node_A.name, str(const_val))
        return new_node

    def compute(self, node, input_vals):
        """Given values of input node, return result of element-wise addition."""
        assert len(input_vals) == 1
        return input_vals[0] + node.const_attr

    def gradient(self, node, output_grad):
        """Given gradient of add node, return gradient contribution to input."""
        return [output_grad]


class SubOp(Op):
    """Op to element-wise add two nodes."""
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s-%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        """Given values of two input nodes, return result of element-wise addition."""
        assert len(input_vals) == 2
        return input_vals[0] - input_vals[1]

    def gradient(self, node, output_grad):
        """Given gradient of add node, return gradient contributions to each input."""
        return [output_grad, (-1) * output_grad]


class SubByConstOp(Op):
    """Op to element-wise add a nodes by a constant."""
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s-%s)" % (node_A.name, str(const_val))
        return new_node

    def compute(self, node, input_vals):
        """Given values of input node, return result of element-wise addition."""
        assert len(input_vals) == 1
        return input_vals[0] - node.const_attr

    def gradient(self, node, output_grad):
        """Given gradient of add node, return gradient contribution to input."""
        return [output_grad]


class MulOp(Op):
    """Op to element-wise multiply two nodes."""
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s*%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        """Given values of two input nodes, return result of element-wise multiplication."""
        """TODO: Your code here"""
        assert len(input_vals) == 2
        return input_vals[0] * input_vals[1]

    def gradient(self, node, output_grad):
        """Given gradient of multiply node, return gradient contributions to each input."""
        """TODO: Your code here"""
        return [output_grad * node.inputs[1], output_grad * node.inputs[0]]


class MulByConstOp(Op):
    """Op to element-wise multiply a nodes by a constant."""
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s*%s)" % (node_A.name, str(const_val))
        return new_node

    def compute(self, node, input_vals):
        """Given values of input node, return result of element-wise multiplication."""
        """TODO: Your code here"""
        assert len(input_vals) == 1
        return input_vals[0] * node.const_attr

    def gradient(self, node, output_grad):
        """Given gradient of multiplication node, return gradient contribution to input."""
        """TODO: Your code here"""
        return [output_grad * node.const_attr]


class DivOp(Op):
    """ Op to element-wise division """
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s/%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        return input_vals[0] / input_vals[1]

    def gradient(self, node, output_grad):
        return [output_grad * oneslike_op(node.inputs[1]) / node.inputs[1],
                (-1) * output_grad * node.inputs[0] / (node.inputs[1] * node.inputs[1])]


class DivByConstOp(Op):
    """ Op to element-wise division by a constant """
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s/%s)" % (node_A.name, str(const_val))
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return input_vals[0] / node.const_attr

    def gradient(self, node, output_grad):
        return [output_grad / node.const_attr]


class MatMulOp(Op):
    """Op to matrix multiply two nodes."""
    def __call__(self, node_A, node_B, trans_A=False, trans_B=False):
        """Create a new node that is the result a matrix multiple of two input nodes.

        Parameters
        ----------
        node_A: lhs of matrix multiply
        node_B: rhs of matrix multiply
        trans_A: whether to transpose node_A
        trans_B: whether to transpose node_B

        Returns
        -------
        Returns a node that is the result a matrix multiple of two input nodes.
        """
        new_node = Op.__call__(self)
        new_node.matmul_attr_trans_A = trans_A
        new_node.matmul_attr_trans_B = trans_B
        new_node.inputs = [node_A, node_B]
        new_node.name = "MatMul(%s,%s,%s,%s)" % (node_A.name, node_B.name,
                                                 str(trans_A), str(trans_B))
        return new_node

    def compute(self, node, input_vals):
        """Given values of input nodes, return result of matrix multiplication."""
        """TODO: Your code here"""
        assert len(input_vals) == 2
        return np.matmul(
                   np.transpose(input_vals[0]) if node.matmul_attr_trans_A \
                                               else input_vals[0],
                   np.transpose(input_vals[1]) if node.matmul_attr_trans_B \
                                               else input_vals[1]
                )


    def gradient(self, node, output_grad):
        """Given gradient of multiply node, return gradient contributions to each input.

        Useful formula: if Y=AB, then dA=dY B^T, dB=A^T dY
        """
        """TODO: Your code here"""
        return [matmul_op(output_grad, node.inputs[1], trans_B=True),
                matmul_op(node.inputs[0], output_grad, trans_A=True)]


class PlaceholderOp(Op):
    """Op to feed value to a nodes."""
    def __call__(self):
        """Creates a variable node."""
        new_node = Op.__call__(self)
        return new_node

    def compute(self, node, input_vals):
        """No compute function since node value is fed directly in Executor."""
        assert False, "placeholder values provided by feed_dict"

    def gradient(self, node, output_grad):
        """No gradient function since node has no inputs."""
        return None


class ZerosLikeOp(Op):
    """Op that represents a constant np.zeros_like."""
    def __call__(self, node_A):
        """Creates a node that represents a np.zeros array of same shape as node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Zeroslike(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        """Returns zeros_like of the same shape as input."""
        assert isinstance(input_vals[0], np.ndarray)
        return np.zeros(input_vals[0].shape)

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]


class OnesLikeOp(Op):
    """Op that represents a constant np.ones_like."""
    def __call__(self, node_A):
        """Creates a node that represents a np.ones array of same shape as node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Oneslike(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        """Returns ones_like of the same shape as input."""
        assert isinstance(input_vals[0], np.ndarray)
        return np.ones(input_vals[0].shape)

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]


class LogOp(Op):
    """ Op that calculates logarithm """
    def __call__(self, node_A):
        """ Creates a node that is the result of the logarithm of input node """
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Log(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        """ Returns logarithm of the node """
        assert len(input_vals) == 1
        return np.log(input_vals[0])

    def gradient(self, node, output_grad):
        return [output_grad * oneslike_op(node.inputs[0]) / node.inputs[0]]


class ExpOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Exp(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return np.exp(input_vals[0])

    def gradient(self, node, output_grad):
        return [output_grad * exp_op(node.inputs[0])]


class SigmoidOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Sigmoid(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return 1 / (1 + np.exp(-input_vals[0]))

    def gradient(self, node, output_grad):
        return [output_grad * sigmoid_op(node.inputs[0]) * sigmoid_op(node.inputs[0]) * exp_op((-1) * node.inputs[0])]


class ReduceSumOp(Op):
    def __call__(self, node_A, reduction_indices=None):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.reduction_indices = reduction_indices
        new_node.name = "ReduceSum(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1

        if node.reduction_indices is None:
            return np.array(np.sum(input_vals[0]))

        return np.sum(input_vals[0], tuple(node.reduction_indices))

    def gradient(self, node, output_grad):
        return [reduce_sum_gradient_op(node.inputs[0], output_grad, node.reduction_indices)]


class ReduceSumGradientOp(Op):
    def __call__(self, node_input, node_output_grad, reduction_indices):
        new_node = Op.__call__(self)
        new_node.inputs = [node_input, node_output_grad]
        new_node.reduction_indices = reduction_indices
        new_node.name = "ReduceSumGradient(%s)" % node_input.name
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        result = input_vals[1].copy()
        if node.reduction_indices is None:
            return result * np.ones(input_vals[0].shape)
        for reduced_ind in sorted(node.reduction_indices):
            result = np.repeat(np.expand_dims(result, axis=reduced_ind),
                               repeats=input_vals[0].shape[reduced_ind],
                               axis=reduced_ind)
        return result

    def gradient(self, node, output_grad):
        return None


class ReduceMeanOp(Op):
    def __call__(self, node_A, reduction_indices=None):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.reduction_indices = reduction_indices
        new_node.name = "ReduceMean(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1

        if node.reduction_indices is None:
            return np.array(np.mean(input_vals[0]))

        return np.mean(input_vals[0], tuple(node.reduction_indices))

    def gradient(self, node, output_grad):
        return [reduce_mean_gradient_op(node.inputs[0],
                                        output_grad,
                                        node.reduction_indices)]


class ReduceMeanGradientOp(Op):
    def __call__(self, node_input, node_output_grad, reduction_indices):
        new_node = Op.__call__(self)
        new_node.inputs = [node_input, node_output_grad]
        new_node.reduction_indices = reduction_indices
        new_node.name = "ReduceMeanGradient(%s)" % node_input.name
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        result = input_vals[1].copy()
        if node.reduction_indices is None:
            reduction_indices = range(len(input_vals[0].shape))
        else:
            reduction_indices = node.reduction_indices

        factor = 1.0
        for reduced_ind in sorted(reduction_indices):
            factor *= input_vals[0].shape[reduced_ind]
            result = np.repeat(np.expand_dims(result, axis=reduced_ind),
                               repeats=input_vals[0].shape[reduced_ind],
                               axis=reduced_ind)
        return result / factor

    def gradient(self, node, output_grad):
        return None


# Create global singletons of operators.
add_op = AddOp()
sub_op = SubOp()
mul_op = MulOp()
add_byconst_op = AddByConstOp()
sub_byconst_op = SubByConstOp()
mul_byconst_op = MulByConstOp()
matmul_op = MatMulOp()
div_op = DivOp()
div_byconst_op = DivByConstOp()
placeholder_op = PlaceholderOp()
oneslike_op = OnesLikeOp()
zeroslike_op = ZerosLikeOp()
log_op = LogOp()
exp_op = ExpOp()
sigmoid_op = SigmoidOp()
reduce_sum_op = ReduceSumOp()
reduce_sum_gradient_op = ReduceSumGradientOp()
reduce_mean_op = ReduceMeanOp()
reduce_mean_gradient_op = ReduceMeanGradientOp()

class Executor:
    """Executor computes values for a given subset of nodes in a computation graph."""
    def __init__(self, eval_node_list):
        """
        Parameters
        ----------
        eval_node_list: list of nodes whose values need to be computed.
        """
        self.eval_node_list = eval_node_list

    def run(self, feed_dict):
        """Computes values of nodes in eval_node_list given computation graph.
        Parameters
        ----------
        feed_dict: list of variable nodes whose values are supplied by user.

        Returns
        -------
        A list of values for nodes in eval_node_list.
        """
        node_to_val_map = dict(feed_dict)

        # Traverse graph in topological sort order and compute values for all nodes.
        topo_order = find_topo_sort(self.eval_node_list)
        """TODO: Your code here"""
        for node in topo_order:
            if isinstance(node.op, PlaceholderOp):
                node_to_val_map[node] = feed_dict[node]
            else:
                node_to_val_map[node] = node.op.compute(
                    node,
                    [node_to_val_map[node] for node in node.inputs]
                )

        # Collect node values.
        node_val_results = [node_to_val_map[node] for node in self.eval_node_list]

        return node_val_results

def gradients(output_node, node_list):
    """Take gradient of output node with respect to each node in node_list.

    Parameters
    ----------
    output_node: output node that we are taking derivative of.
    node_list: list of nodes that we are taking derivative wrt.

    Returns
    -------
    A list of gradient values, one for each node in node_list respectively.

    """

    # a map from node to a list of gradient contributions from each output node
    node_to_output_grads_list = {}

    # Special note on initializing gradient of output_node as oneslike_op(output_node):
    # We are really taking a derivative of the scalar reduce_sum(output_node)
    # instead of the vector output_node. But this is the common case for loss function.
    node_to_output_grads_list[output_node] = [oneslike_op(output_node)]

    # a map from node to the gradient of that node
    node_to_output_grad = {}

    # Traverse graph in reverse topological order given the output_node that we are taking gradient wrt.
    reverse_topo_order = reversed(find_topo_sort([output_node]))

    """TODO: Your code here"""
    for node in reverse_topo_order:
        grad_ = sum(node_to_output_grads_list[node])
        node_to_output_grad[node] = grad_
        if len(node.inputs) == 0:
            continue
        input_grad_ = node.op.gradient(node, grad_)
        for input_, input_grad_ in zip(node.inputs, input_grad_):
            if input_ not in node_to_output_grads_list:
                node_to_output_grads_list[input_] = []
            node_to_output_grads_list[input_].append(input_grad_)

    # Collect results for gradients requested.
    grad_node_list = [node_to_output_grad[node] for node in node_list]
    return grad_node_list

##############################
####### Helper Methods #######
##############################

def find_topo_sort(node_list):
    """Given a list of nodes, return a topological sort list of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.

    """
    visited = set()
    topo_order = []
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order

def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    if node in visited:
        return
    visited.add(node)
    for n in node.inputs:
        topo_sort_dfs(n, visited, topo_order)
    topo_order.append(node)

def sum_node_list(node_list):
    """Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
    from operator import add
    from functools import reduce
    return reduce(add, node_list)
