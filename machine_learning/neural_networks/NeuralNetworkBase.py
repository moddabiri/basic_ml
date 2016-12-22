import numpy as np
import ml_util.math_helpers as mh

from math import sqrt
from abc import ABCMeta, abstractmethod
from machine_learning.CostMinimizerBase import CostMinimizerBase

class NeuralNetworkBase(CostMinimizerBase):
    """description of class"""

    #__metaclass__ = ABCMeta

    def __init__(self, hidden_layer_sizes, classes, iterations, alpha, doNormalize = False, lambdaRate = 0.0, biasValue = 1):
        self.no_of_layers = len(hidden_layer_sizes) + 2
        self.biasValue = biasValue
        self.classes = classes

        #Build the structure of layer-set
        self.layers = [None] * (len(hidden_layer_sizes) + 2)
        self.layer_sizes = [None] + hidden_layer_sizes + [None]

        #Bulid the structure of hidden layers
        for i in range(len(hidden_layer_sizes)):
            #+1 for bias value
            layer_size = hidden_layer_sizes[i] + 1
            self.layers[i + 1] = [None] * layer_size

        return super().__init__(iterations, alpha, True, doNormalize, lambdaRate, mapping=None, labelCount=1)

    #The bias value will be included in input and hidden layers {integer}
    biasValue = 1

    #Total number of layers in nn including input, hidden and output layer {integer}
    no_of_layers = 0

    #The layers in nn including input, hidden and output layer {3d array: no_of_layers x m x [layer_size [+ 1 except for output layer]]}
    layers = []

    #The original sizes of layers in nn including input, hidden and output layer (does not include bias) {2d array: no_of_layers x 1}
    layer_sizes = []

    #Theta of all layers for each example {4d array: m x [no_of_layers - 1] x matrix(l_in x l_out)}
    theta_layers = []

    def initialize_properties(self):
        #For classification problems
        no_of_classes = len(self.classes)

        #Bias is always included in nn in input and hidden layers
        self.n = self.n + 1

        #Add bias to input layer
        self.x = np.vstack((np.ones((1, self.m), dtype=np.float) * self.biasValue, self.x))

        #Build up the layers of the nn
        self.layers[0] = self.x.transpose()
        self.layer_sizes[0] = self.n - 1
        self.layers[self.no_of_layers - 1] = self.y
        self.layer_sizes[self.no_of_layers - 1] = no_of_classes
        self.theta_layers = [None] * (self.m)

        for example_indx in range (self.m):
            example_theta = [None] * (self.no_of_layers - 1)

            #For each example randomize the initial theta for each layer
            for layer_indx in range(self.no_of_layers - 1):
                l_in = self.layer_sizes[layer_indx]
                l_out = self.layer_sizes[layer_indx + 1]
                example_theta[layer_indx] = self.random_initialize_weights(l_in, l_out)

            self.theta_layers[example_indx] = example_theta

    def random_initialize_weights(self, l_in, l_out):
        INIT_EPSILON = sqrt(6) / sqrt(l_out + l_in)
        return np.random.rand(l_in + 1, l_out) * (2 * INIT_EPSILON) - INIT_EPSILON

    def compute_cost(self, theta, classIndex):
        for example_indx in range(self.m):
            self.forward_propagate(example_indx)
            self.backward_propagate(example_indx)

    def forward_propagate(self, example_indx):
        theta_layers = self.theta_layers[example_indx]

        for layer_indx in range(self.no_of_layers - 1):
            x = self.layers[layer_indx][example_indx]
            theta = theta_layers[layer_indx]
            self.layers[layer_indx + 1][example_indx] = mh.sigmoid(np.dot(theta, x))
            #TODO: Continue from here
    
    def backward_propagate(self):
        #TODO: Finish
        pass       