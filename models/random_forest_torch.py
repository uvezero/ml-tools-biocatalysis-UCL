import torch
import random
from math import sqrt, log
import numpy as np

# REF:
# https://github.com/ValentinFigue/Sklearn_PyTorch/tree/master
# Valentin Figue

class DecisionNode:
    """
    A decision node for building a binary tree, used in decision tree algorithms.
    """

    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        self.col = col  # Column index for the attribute/feature
        self.value = value  # Value to split the column
        self.results = results  # Stores predictions at the leaf node
        self.tb = tb  # True branch (subtree)
        self.fb = fb  # False branch (subtree)




class TorchDecisionTreeRegressor(torch.nn.Module):
    """
    Torch decision tree object used to solve regression problem. This object implements the fitting and prediction
    function which can be used with torch tensors. The binary tree is based on
    :class:`Sklearn_PyTorch.decision_node.DecisionNode` which are built during the :func:`fit` and called recursively during the
    :func:`predict`.

    Args:
        max_depth (:class:`int`): The maximum depth which corresponds to the maximum successive number of
            :class:`Sklearn_PyTorch.decision_node.DecisionNode`.

    """
    def __init__(self, max_depth=-1):
        self._root_node = None
        self.max_depth = max_depth

    def fit(self, vectors, values, criterion=None):
        """
        Function which must be used after the initialisation to fit the binary tree and build the successive
        :class:`Sklearn_PyTorch.decision_node.DecisionNode` to solve a specific regression problem.

        Args:
            vectors(:class:`torch.FloatTensor`): Vectors tensor used to fit the decision tree. It represents the data
                and must correspond to the following shape (num_vectors, num_dimensions_vectors).
            values(:class:`torch.FloatTensor`): Values tensor used to fit the decision tree. It represents the values
                associated to each vectors and must correspond to the following shape (num_vectors,
                num_dimensions_values).
            criterion(:class:`function`): Optional function used to optimize the splitting for each
                :class:`Sklearn_PyTorch.decision_node.DecisionNode`. If none given, the variance function is used.
        """
        if len(vectors) < 1:
            raise ValueError("Not enough samples in the given dataset")
        if len(vectors) != len(values):
            raise ValueError("Labels and data vectors must have the same number of elements")
        if not criterion:
            criterion = variance #variance #impement cost function
            
        self._root_node = self._build_tree(vectors, values, criterion, self.max_depth)

    def _build_tree(self, vectors, values, func, depth):
        """
        Private recursive function used to build the tree.
        """
        if len(vectors) == 0:
            return DecisionNode()
        if depth == 0:
            return DecisionNode(results=mean(values))
 
        current_score = func(values)
        best_gain = 0.0
        best_criteria = None
        best_sets = None
        column_count = len(vectors[0])

        for col in range(0, column_count):
            column_values = {}
            for vector in vectors:
                column_values[vector[col]] = 1 # dummy values
            for value in column_values.keys():
                vectors_set_1, values_set_1, vectors_set_2, values_set_2 = divide_set(vectors, values, col, value)

                p = float(len(vectors_set_1)) / len(vectors)
                gain = current_score - p * func(values_set_1) - (1 - p) * func(vectors_set_2)
                if gain > best_gain and len(vectors_set_1) > 0 and len(vectors_set_2) > 0:
                    best_gain = gain
                    best_criteria = (col, value)
                    best_sets = ((vectors_set_1,values_set_1), (vectors_set_2,values_set_2))

        if best_gain > 0:
            true_branch = self._build_tree(best_sets[0][0], best_sets[0][1], func, depth - 1)
            false_branch = self._build_tree(best_sets[1][0], best_sets[1][1], func, depth - 1)
            return DecisionNode(col=best_criteria[0],
                                value=best_criteria[1],
                                tb=true_branch, fb=false_branch)
        else:
            return DecisionNode(results=mean(values))

    def predict(self, vector):
        """
        Function which must be used after the the fitting of the binary tree. It calls recursively the different
        :class:`Sklearn_PyTorch.decision_node.DecisionNode` to regress the vector.

        Args:
            vector(:class:`torch.FloatTensor`): Vectors tensor which must be regressed. It represents the data
                and must correspond to the following shape (num_dimensions).

        Returns:
            :class:`torch.FloatTensor`: Tensor which corresponds to the value regressed by the binary tree.

        """
        return self._regress(vector, self._root_node)

    def _regress(self, vector, node):
        """
        Private recursive function used to regress on the tree.
        """
        if node.results is not None:
            return node.results
        else:
            if split_function(vector, node.col, node.value):
                branch = node.tb
            else:
                branch = node.fb

            return self._regress(vector, branch)


def unique_counts(labels):
    """
    Unique count function used to count labels.
    """
    results = {}
    for label in labels:
        value = label.item()
        if value not in results.keys():
            results[value] = 0
        results[value] += 1
    return results


def divide_set(vectors, labels, column, value):
    """
    Divide the sets into two different sets along a specific dimension and value.
    """
    set_1 = [(vector, label) for vector, label in zip(vectors, labels) if split_function(vector, column, value)]
    set_2 = [(vector, label) for vector, label in zip(vectors, labels) if not split_function(vector, column, value)]

    vectors_set_1 = [element[0] for element in set_1]
    vectors_set_2 = [element[0] for element in set_2]
    label_set_1 = [element[1] for element in set_1]
    label_set_2 = [element[1] for element in set_2]

    return vectors_set_1, label_set_1, vectors_set_2, label_set_2


def split_function(vector, column, value):
    """
    Split function
    """
    return vector[column] >= value


def log2(x):
    """
    Log2 function
    """
    return log(x) / log(2)


def sample_vectors(vectors, labels, nb_samples):
    """
    Sample vectors and labels uniformly.
    """
    sampled_indices = torch.LongTensor(random.sample(range(len(vectors)), nb_samples))
    sampled_vectors = torch.index_select(vectors,0, sampled_indices)
    sampled_labels = torch.index_select(labels,0, sampled_indices)

    return sampled_vectors, sampled_labels


def sample_dimensions(vectors):
    """
    Sample vectors along dimension uniformly.
    """
    sample_dimension = torch.LongTensor(random.sample(range(len(vectors[0])), int(sqrt(len(vectors[0])))))

    return sample_dimension


def entropy(labels):
    """
    Entropy function.
    """
    results = unique_counts(labels)
    ent = 0.0
    for r in results.keys():
        p = float(results[r]) / len(labels)
        ent = ent - p * log2(p)
    return ent

##############################################################################################
#                                                                                            #
#                                       LOSS FUNCTIONS                                       #
#                                                                                            #
##############################################################################################

def mean_squared_error(values):
    """
    Compute the mean squared error of the values.
    """
    if len(values) == 0:
        return 0.0

    mean_value = mean(values)
    mse = 0.0

    for value in values:
        # Check if value is a tensor with more than one element
        if isinstance(value, torch.Tensor) and value.nelement() > 1:
            squared_diffs = (value - mean_value) ** 2
            mse += torch.sum(squared_diffs).item()
        else:
            mse += ((value - mean_value) ** 2).item()

    mse /= len(values)
    return mse


def variance(values):
    """
    Variance function.
    """
    mean_value = mean(values)
    var = 0.0
    for value in values:
        var = var + torch.sum(torch.sqrt(torch.pow(value-mean_value,2))).item()/len(values)
    return var

def low_loss(values):
    '''
    Custom loss function to bias results towards smaller barriers.
    '''
    epsilon = 1e-7
    mean_value = mean(values)
    custom_MSE = 0.0
    for val in values:
        weight = 1.0 / (val + epsilon)
        squared_errors = torch.sum(torch.pow(weight * (val - mean_value),2)).item()/(len(values))
        custom_MSE = custom_MSE + squared_errors
    return sqrt(custom_MSE)

def exponential_decay_variance(y_true, decay_rate=0.1, epsilon=1e-6):
    decay_weights = torch.exp(-decay_rate * y_true)
    weighted_mean = torch.sum(decay_weights * y_true) / torch.sum(decay_weights)
    variance = torch.sum(decay_weights * ((y_true - weighted_mean) ** 2)) / torch.sum(decay_weights)
    return variance


def inverse_weighted_variance(values, epsilon=1e-6):
    # Ensure y_true is a PyTorch tensor    
    weights = 1 / (values + epsilon)
    weighted_mean = torch.sum(weights * values) / torch.sum(weights)
    variance = torch.sum(weights * ((values - weighted_mean) ** 2)) / torch.sum(weights)
    return variance

def mean(values):
    """
    Mean function.
    """
    m = 0.0
    for value in values:
        m = m + value/len(values)
    return m

    
class TorchRandomForestRegressor(torch.nn.Module):
    """
    Random Forest Regressor implemented using PyTorch.
    """

    def __init__(self, nb_trees, nb_samples, max_depth=-1, bootstrap=True):
        self.trees = []
        self.nb_trees = nb_trees
        self.nb_samples = nb_samples
        self.max_depth = max_depth
        self.trees_features = []
        self.bootstrap = bootstrap

    def fit(self, vectors, values):
        """
        Fits the random forest model to the given data.
        """
        
        for _ in range(self.nb_trees):
            tree = TorchDecisionTreeRegressor(self.max_depth)
            list_features = sample_dimensions(vectors)
            self.trees_features.append(list_features)
            if self.bootstrap:
                sampled_vectors, sample_labels = sample_vectors(vectors, values, self.nb_samples)
                sampled_featured_vectors = torch.index_select(sampled_vectors, 1, list_features)
                tree.fit(sampled_featured_vectors, sample_labels)
            else:
                sampled_featured_vectors = torch.index_select(vectors, 1, list_features)
                tree.fit(sampled_featured_vectors, values)
            self.trees.append(tree)
        
        
        '''self._validate_input(vectors, values)

        for _ in range(self.nb_trees):
            tree = TorchDecisionTreeRegressor(self.max_depth)
            sampled_vectors, sampled_values = self._sample_data(vectors, values)
            tree.fit(sampled_vectors, sampled_values)
            self.trees.append(tree)'''
            
    def predict(self, vector):
        """
        Function which must be used after the the fitting of the random forest. It calls recursively the different
        :class:`Sklearn_PyTorch.binary_tree.TorchDecisionTreeRegressor` to regress the vector.

        Args:
            vector(:class:`torch.FloatTensor`): Vectors tensor which must be regressed. It represents the data
                and must correspond to the following shape (num_dimensions).

        Returns:
            :class:`torch.FloatTensor`: Tensor which corresponds to the value regressed by the random forest.

        """
        predictions_sum = 0
        for tree, index_features in zip(self.trees, self.trees_features):
            sampled_vector = torch.index_select(vector, 0, index_features)
            predictions_sum += tree.predict(sampled_vector)

        return predictions_sum/len(self.trees)
