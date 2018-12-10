"""Computation graph node types

Nodes must implement the following methods:
__init__   - initialize node
forward    - (step 1 of backprop) retrieve output ("out") of predecessor nodes (if
             applicable), update own output ("out"), and set gradient ("d_out") to zero
backward   - (step 2 of backprop), assumes that forward pass has run before.
             Also assumes that backward has been called on all of the node's
             successor nodes, so that self.d_out contains the
             gradient of the graph output with respect to the node output.
             Backward computes summands of the derivative of graph output with
             respect to the inputs of the node, corresponding to paths through the graph
             that go from the node's input through the node to the graph's output.
             These summands are added to the input node's d_out array.
get_predecessors - return a list of the node's parents

Nodes must furthermore have a the following attributes:
node_name  - node's name (a string)
out      - node's output
d_out    - derivative of graph output w.r.t. node output

License: Creative Commons Attribution 4.0 International License

Acknowledge: This computation graph framework was designed by
Philipp Meerkamp, Pierre Garapon, and David Rosenberg. This is a python3 implementation
by Yi Zhou after taking David Rosenberg's DS-GA 1003 Machine Learning Course at NYU Data
Science Center and for the sake of CS-GY 6643 Computer Vision Project.

Author: Yi Zhou
"""


import numpy as np


class ValueNode:
    """Computation graph node having no input but simply holding a value"""

    def __init__(self, node_name):
        self.node_name = node_name
        self.out = None
        self.d_out = None

    def forward(self):
        self.d_out = np.zeros(self.out.shape)
        return self.out

    def backward(self):
        pass

    def get_predecessors(self):
        return []


class VectorScalarAffineNode:
    """ Node computing an affine function mapping a vector to a scalar."""

    def __init__(self, x, w, b, node_name):
        """
        Parameters:
        x: node for which x.out is a 1D numpy array
        w: node for which w.out is a 1D numpy array of same size as x.out
        b: node for which b.out is a numpy scalar (i.e. 0dim array)
        node_name: node's name (a string)
        """
        self.node_name = node_name
        self.out = None
        self.d_out = None
        self.x = x
        self.w = w
        self.b = b

    def forward(self):
        self.out = np.dot(self.x.out, self.w.out) + self.b.out
        self.d_out = np.zeros(self.out.shape)
        return self.out

    def backward(self):
        d_x = self.d_out * self.w.out
        d_w = self.d_out * self.x.out
        d_b = self.d_out
        self.x.d_out += d_x
        self.w.d_out += d_w
        self.b.d_out += d_b

    def get_predecessors(self):
        return [self.x, self.w, self.b]


class SquaredL2DistanceNode:
    """ Node computing L2 distance (sum of square differences) between 2 arrays."""

    def __init__(self, a, b, node_name):
        """
        Parameters:
        a: node for which a.out is a numpy array
        b: node for which b.out is a numpy array of same shape as a.out
        node_name: node's name (a string)
        """
        self.node_name = node_name
        self.out = None
        self.d_out = None
        self.a = a
        self.b = b
        # Variable for caching values between forward and backward
        self.a_minus_b = None

    def forward(self):
        self.a_minus_b = self.a.out - self.b.out
        self.out = np.sum(self.a_minus_b ** 2)
        self.d_out = np.zeros(self.out.shape)
        return self.out

    def backward(self):
        d_a = self.d_out * 2 * self.a_minus_b
        d_b = self.d_out * 2 * self.a_minus_b
        self.a.d_out += d_a
        self.b.d_out -= d_b
        return self.d_out

    def get_predecessors(self):
        return [self.a, self.b]


class L2NormPenaltyNode:
    """ Node computing l2_reg * ||w||^2 for scalars l2_reg and vector w"""

    def __init__(self, l2_reg, w, node_name):
        """
        Parameters:
        l2_reg: a scalar value >=0 (not a node)
        w: a node for which w.out is a numpy vector
        node_name: node's name (a string)
        """
        self.node_name = node_name
        self.out = None
        self.d_out = None
        self.l2_reg = np.array(l2_reg)
        self.w = w

    def forward(self):
        self.out = np.sum(self.w.out ** 2) * self.l2_reg
        self.d_out = np.zeros(self.out.shape)
        return self.out

    def backward(self):
        d_w = self.d_out * 2 * self.w.out * self.l2_reg
        self.w.d_out += d_w
        return self.d_out

    def get_predecessors(self):
        return [self.w]


class SumNode:
    """ Node computing a + b, for numpy arrays a and b"""

    def __init__(self, a, b, node_name):
        """
        Parameters:
        a: node for which a.out is a numpy array
        b: node for which b.out is a numpy array of the same shape as a
        node_name: node's name (a string)
        """
        self.node_name = node_name
        self.out = None
        self.d_out = None
        self.a = a
        self.b = b

    def forward(self):
        self.out = self.a.out + self.b.out
        self.d_out = np.zeros(self.out.shape)
        return self.out

    def backward(self):
        d_a = self.d_out
        d_b = self.d_out
        self.a.d_out += d_a
        self.b.d_out += d_b
        return self.d_out

    def get_predecessors(self):
        return [self.a, self.b]


class AffineNode:
    """Node implementing affine transformation (W,x,b)-->Wx+b, where W is a matrix,
    and x and b are vectors
    """

    def __init__(self, W, x, b, node_name):
        """
        Parameters:
        W: node for which W.out is a numpy array of shape (m,d)
        x: node for which x.out is a numpy array of shape (d)
        b: node for which b.out is a numpy array of shape (m) (i.e. vector of length m)
        """
        self.node_name = node_name
        self.out = None
        self.d_out = None
        self.W = W
        self.x = x
        self.b = b

    def forward(self):
        self.out = self.W.out.dot(self.x.out) + self.b.out
        self.d_out = np.zeros(self.out.shape)
        return self.out

    def backward(self):
        m = self.d_out.shape[0]
        d_b = self.d_out
        d_W = self.d_out.reshape((m, 1)) * self.x.out
        d_x = self.W.out.T.dot(self.d_out)
        self.b.d_out += d_b
        self.W.d_out += d_W
        self.x.d_out += d_x
        return self.d_out

    def get_predecessors(self):
        return [self.b, self.W, self.x]


class TanhNode:
    """Node tanh(a), where tanh is applied element-wise to the array a
    """

    def __init__(self, a, node_name):
        """
        Parameters:
        a: node for which a.out is a numpy array
        """
        self.node_name = node_name
        self.out = None
        self.d_out = None
        self.a = a

    def forward(self):
        self.out = np.tanh(self.a.out)
        self.d_out = np.zeros(self.out.shape)
        return self.out

    def backward(self):
        d_a = self.d_out * (np.ones(self.out.shape) - self.out ** 2)
        self.a.d_out += d_a
        return self.d_out

    def get_predecessors(self):
        return [self.a]


class SigmoidNode:
    """Node sigmoid(a), where sigmoid is applied element-wise to the array a
    """

    def __init__(self, a, node_name):
        """
        Parameters:
        a: node for which a.out is a numpy array
        """
        self.node_name = node_name
        self.out = None
        self.d_out = None
        self.a = a

    def forward(self):
        # print("---")
        # print("debug message, input in SigmoidNode: a", self.a.out)
        self.out = 1 / (1 + np.exp(-self.a.out))
        self.d_out = np.zeros(self.out.shape)
        # print("debug message, output in SigmoidNode:", self.out)
        return self.out

    def backward(self):
        d_a = self.d_out * (1 - self.out) * self.out
        self.a.d_out += d_a
        return self.d_out

    def get_predecessors(self):
        return [self.a]


class BinaryCrossEntropyLossNode:
    """Node for compute Binary Cross Entropy element-wise to the arrays a, y
       Loss(a,y) = -yln(a)-(1-y)ln(1-a), y should in {0, 1}, a should be a positive value in (0,1)
    """

    def __init__(self, a, y, node_name):
        """
        Parameters:
        a: node for which a.out is a numpy array storing probas
        y: a ValueNode for which a.out is a numpy array storing outcomes
        """
        self.node_name = node_name
        self.out = None
        self.d_out = None
        self.a = a
        self.y = y

    def forward(self):
        # print("---")
        # print("debug message, input in BinaryCrossEntropyLossNode: a", self.a.out)
        # print("debug message, input in BinaryCrossEntropyLossNode: y", self.y.out)
        self.out = -self.y.out * np.log(self.a.out) - (1 - self.y.out) * np.log(1 - self.a.out)
        self.d_out = np.zeros(self.out.shape)
        # print("debug message, output in BinaryCrossEntropyLossNode:", self.out)
        return self.out

    def backward(self):
        d_a = self.d_out * (-self.y.out / self.a.out + (1 - self.y.out) / (1 - self.a.out))
        self.a.d_out += d_a
        return self.d_out

    def get_predecessors(self):
        return [self.a, self.y]