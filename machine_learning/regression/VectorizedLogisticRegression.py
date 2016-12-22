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
from machine_learning.regression.LogisticRegression import LogisticRegression
from functools import lru_cache as cache_result

class VectorizedLogisticRegression(LogisticRegression):
    """One vs. all classifier"""

    def __init__(self, iterations, alpha, classes, labelCount = 1, includeBias = False, doNormalize = False, lambdaRate = 0.0, mapping = None):        
        if not isinstance(classes, list):
            raise ValueError("Vectorized regression for one vs all requires classes paramter as an array of values for each class.")

        self.classes = classes
        noOfClasses = len(classes)
        self.costs = [None]*noOfClasses

        #In this scenario, y is given as a mx1 array of labeled items with different classes, the model is defined to intake more than one set of labeled data
        return super().__init__(iterations, alpha, includeBias, doNormalize, lambdaRate, mapping, labelCount)
        
    @cache_result(maxsize=None)
    def get_labeled_set(self, classIndex = 0):
        #In this scenario, y is given as a mx1 array of labeled items with different classes
        y = super().get_labeled_set()

        c = self.classes[classIndex]
        labeled = [1 if val == c else 0 for val in y]
        return np.matrix(labeled).transpose()

    def predict(self):
        z = np.dot(self.theta, self.x)
        h = sigmoid(z)
        #TODO: It should get the maximum probability for each class and set that as the result
        raise NotImplementedError()


        