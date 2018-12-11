"""Models Collection

License: Creative Commons Attribution 4.0 International License

Author: Yi Zhou
"""

import numpy as np
import simplenn.nodes as nodes
import simplenn.graph as graph
from simplenn.utility import timer


class MLPBinaryClassifier:
    """
    Two-layer perceptron binary classifier.
    Using tanh as a hidden layer activation function.
    """
    def __init__(self, num_hidden_units=10, step_size=.005, init_param_scale=0.01, max_num_epochs=500,
                 tolerance=0.000001, early_stop=False, verbose=False, activation="relu"):
        self.num_hidden_units = num_hidden_units
        self.init_param_scale = init_param_scale
        self.max_num_epochs = max_num_epochs
        self.step_size = step_size
        self.tolerance = tolerance
        self.early_stop = early_stop
        self.verbose = verbose
        self.activation = activation
        
        # Build computation graph
        self.x = nodes.ValueNode(node_name="x")  # to hold a vector input
        self.y = nodes.ValueNode(node_name="y")  # to hold a scalar response
        self.W1 = nodes.ValueNode(node_name="W1")  # to hold the parameter matrix
        self.b1 = nodes.ValueNode(node_name="b1")  # to hold the parameter vector
        self.L = nodes.AffineNode(self.W1, self.x, self.b1, node_name="L")
        if self.activation == "tanh": self.h = nodes.TanhNode(self.L, node_name="h")
        elif self.activation == "relu": self.h = nodes.ReluNode(self.L, node_name ="h")
        else: raise ValueError("Un-supported Activation Type")
        self.W2 = nodes.ValueNode(node_name="W2")  # to hold the parameter vector
        self.b2 = nodes.ValueNode(node_name="b2")  # to hold the parameter scalar
        self.a = nodes.VectorScalarAffineNode(self.W2, self.h, self.b2, node_name="a")

        self.inputs = [self.x]
        self.outcomes = [self.y]
        self.parameters = [self.W1, self.b1, self.W2, self.b2]
        self.prediction = nodes.SigmoidNode(self.a, node_name="prediction")
        self.objective = nodes.BinaryCrossEntropyLossNode(self.prediction, self.y, node_name="objective")

        self.graph = graph.ComputationGraph(self.inputs, self.outcomes,
                                            self.parameters, self.prediction,
                                            self.objective)

    @timer
    def fit(self, X, y):
        num_instances, num_features = X.shape
        y = y.reshape(-1)
        s = self.init_param_scale
        init_values = {"W1": s * np.random.standard_normal(size=(self.num_hidden_units, num_features)),
                       "b1": s * np.random.standard_normal(size=(self.num_hidden_units,)),
                       "W2": s * np.random.standard_normal(size=(self.num_hidden_units,)),
                       "b2": s * np.array(np.random.randn())}

        self.graph.set_parameters(init_values)
        last_train_loss = None
        for epoch in range(self.max_num_epochs):
            shuffle = np.random.permutation(num_instances)
            epoch_obj_tot = 0.0
            for j in shuffle:
                obj, grads = self.graph.get_gradients(input_values={"x": X[j]}, outcome_values={"y": y[j]})
                epoch_obj_tot += obj
                # Take step in negative gradient direction
                steps = {}
                for param_name in grads:
                    steps[param_name] = -self.step_size * grads[param_name]
                    self.graph.increment_parameters(steps)

            if epoch % 5 == 0:
                a = self.predict_proba(X)
                train_loss = np.sum(-y * np.log(a) - (1 - y) * np.log(1 - a)) / num_instances
                if self.verbose:
                    print("Epoch ", epoch, ": Ave objective=", epoch_obj_tot / num_instances, " Ave training loss= ",
                          train_loss)
                if self.early_stop and last_train_loss is not None and \
                        np.abs(train_loss-last_train_loss) < self.tolerance:
                     print("☺️  early_stop triggered! Training stops due to this model already meet the expectation.")
                     break
                last_train_loss = train_loss

    def predict_proba(self, X):
        try:
            getattr(self, "graph")
        except AttributeError:
            raise RuntimeError("You must train classifier before predicting data!")
        num_instances = X.shape[0]
        preds = np.zeros(num_instances)
        for j in range(num_instances):
            preds[j] = self.graph.get_prediction(input_values={"x": X[j]})
        return preds

    @timer
    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        labels = np.zeros(probs.shape, dtype="int64")
        for i, p in enumerate(probs):
            if p >= threshold:
                labels[i] = 1
        return labels
