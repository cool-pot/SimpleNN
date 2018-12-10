"""Some common utilities can be used in the repo

License: Creative Commons Attribution 4.0 International License

Author: Yi Zhou
"""
import numpy as np
import time
import datetime

# decorator utilities


def timer(func):
    def wrapper(*arg, **kw):
        func_name = func.__name__
        print('🕛 [%s] Start: %s' % (func_name, '{0:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())))
        t1 = time.time()
        res = func(*arg, **kw)
        t2 = time.time()
        print('🕒 [%s] Finish: %s' % (func_name, '{0:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())))
        print('🕒 [%s] Finish in %.5f seconds' % (func_name, t2-t1))
        return res
    return wrapper

# Evaluation utilities


def precision(prediction, truth):
    return np.sum(prediction == truth) / prediction.size


def recall(prediction, truth):
    return np.sum(prediction * truth) / np.sum(truth)


def evaluate_binary_classification(prediction, truth, expected_precision=0.5, expected_recall=0.5):
    p = precision(prediction, truth)
    r = recall(prediction, truth)
    p_sign = "✅" if p > expected_precision else "❌"
    r_sign = "✅" if r > expected_recall else "❌"
    print("Binary classification performance report:")
    print(p_sign, "Precision: %.5f" % p)
    print(r_sign, "Recall %.5f" % r)


# Computation graph utilities


def sort_topological(sink):
    """Returns a list of the sink node and all its ancestors in topologically sorted order."""
    L = [] # Empty list that will contain the sorted nodes
    T = set() # Set of temporarily marked nodes
    P = set() # Set of permanently marked nodes

    def visit(node):
        if node in P:
            return
        if node in T:
            raise ValueError('Your graph is not a DAG!')
        T.add(node) # mark node temporarily
        for predecessor in node.get_predecessors():
            visit(predecessor)
        P.add(node) # mark node permanently
        L.append(node)

    visit(sink)
    return L


def forward_graph(graph_output_node, node_list=None):
    # If node_list is not None, it should be sort_topological(graph_output_node)
    if node_list is None:
        node_list = sort_topological(graph_output_node)
    for node in node_list:
        out = node.forward()
    return out


def backward_graph(graph_output_node, node_list=None):
    """
    If node_list is not None, it should be the reverse of sort_topological(graph_output_node).
    Assumes that forward_graph has already been called on graph_output_node.
    Sets d_out of each node to the appropriate derivative.
    """
    if node_list is None:
        node_list = sort_topological(graph_output_node)
        node_list.reverse()

    graph_output_node.d_out = np.array(1) # Derivative of graph output w.r.t. itself is 1

    for node in node_list:
        node.backward()
