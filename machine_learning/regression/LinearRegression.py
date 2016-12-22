__author__ = "Mohammad Dabiri"
__copyright__ = "Free to use, copy and modify"
__credits__ = ["Mohammad Dabiri"]
__license__ = "MIT Licence"
__version__ = "0.0.1"
__maintainer__ = "Mohammad Dabiri"
__email__ = "moddabiri@yahoo.com"

import numpy as np
import importlib
import os.path

if not importlib.find_loader('matplotlib') is None:
    import matplotlib.pyplot as plt
else:
    print("WARNING! matplotlicb package was not installed on the vm. Plotting functionalities will not work.")

from machine_learning.CostMinimizerBase import CostMinimizerBase

class LinearRegression(CostMinimizerBase):

    def __init__(self, iterations, alpha, includeBias = False, doNormalize = False, lambdaRate = 0.0, mapping = None):
        return super().__init__(iterations, alpha, includeBias, doNormalize, lambdaRate, mapping, labelCount=1)

    def compute_distances(self, theta, y):
        return np.subtract(theta.dot(self.x), y.transpose())
    
    def compute_cost(self, theta, classIndex):
        y = self.get_labeled_set(classIndex)
        distances = self.compute_distances(theta, y)
        costMatrix = np.power(distances, 2)
        cost = (1 / (2 * float(self.m))) * np.sum(costMatrix)

        if (self.lambdaRate > 0.0):
            cost = self.regularize_cost(cost, theta)

        return cost

    def compute_grads(self, theta, classIndex):
        y = self.get_labeled_set(classIndex)
        distances = self.compute_distances(theta, y)
        grads = distances.dot(self.x.transpose())

        if (self.lambdaRate > 0.0):
            grads = self.regularize_grads(grads, theta)

        return grads