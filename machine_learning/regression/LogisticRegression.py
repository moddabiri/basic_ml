__author__ = "Mohammad Dabiri"
__copyright__ = "Free to use, copy and modify"
__credits__ = ["Mohammad Dabiri"]
__license__ = "MIT Licence"
__version__ = "0.0.1"
__maintainer__ = "Mohammad Dabiri"
__email__ = "moddabiri@yahoo.com"

import numpy as np
import importlib
import math
import os.path

if not importlib.find_loader('matplotlib') is None:
    import matplotlib.pyplot as plt
else:
    print("WARNING! matplotlicb package was not installed on the vm. Plotting functionalities will not work.")

from ml_util.math_helpers import sigmoid, get_polynomial, cross_multiplication
from machine_learning.CostMinimizerBase import CostMinimizerBase

class LogisticRegression(CostMinimizerBase):
    """description of class"""

    def __init__(self, iterations, alpha, includeBias = False, doNormalize = False, lambdaRate = 0.0, mapping = None, labelCount=1):
        return super().__init__(iterations, alpha, includeBias, doNormalize, lambdaRate, mapping, labelCount=1)
    
    def compute_cost(self, theta, classIndex):
        y = self.get_labeled_set(classIndex)
        z = np.dot(theta, self.x)
        h = sigmoid(z)
        cost = (-1 * (1.0 / float(self.m)) * (np.dot(np.log(h), y) + np.dot(np.log(1 - h), (1 - y)))).item(0, 0)
        
        if (self.lambdaRate > 0.0):
            cost = self.regularize_cost(cost, theta)

        return cost

    def compute_grads(self, theta, classIndex):
        y = self.get_labeled_set(classIndex)
        z = np.dot(theta, self.x)
        h = sigmoid(z)
        grads = (1.0 / float(self.m)) * np.dot((h - y.transpose()), self.x.transpose())

        if (self.lambdaRate > 0.0):
            grads = self.regularize_grads(grads, theta)

        return grads

    def predict(self):
        z = np.dot(self.theta, self.x)
        h = sigmoid(z)
        return [0.0 if x < 0.5 else 1.0 for x in np.asarray(h)[0,:]]