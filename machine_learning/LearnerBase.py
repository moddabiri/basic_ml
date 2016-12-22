import numpy as np
import importlib
import math
import os
import sys
import ml_util.math_helpers as mh
import types
import mmap

from abc import ABCMeta, abstractmethod
from ml_util.functional_utility import check_none
from functools import lru_cache as cache_result
from machine_learning.CostMinimizationAlgorithms import CostMinimizationAlgorithms

if not importlib.find_loader('matplotlib') is None:
    import matplotlib.pyplot as plt
else:
    print("WARNING! matplotlicb package was not installed on the vm. Plotting functionalities will not work.")

class LearnerBase():
    __metaclass__ = ABCMeta

    def __init__(self, includeBias = False, doNormalize = False, lambdaRate = 0.0, mapping = None, labelCount=1):
        check_none(includeBias=includeBias, doNormalize=doNormalize, lambdaRate=lambdaRate)

        if (lambdaRate < 0.0):
            raise ValueError("Lamda must not be less than 0.0")

        if not mapping is None and (not isinstance(mapping, tuple) or not isinstance(mapping[0], list) or not mapping[1] > 0):
            raise ValueError("Feature mapping degree must be more than 0")

        self.includeBias = includeBias
        self.doNormalize = doNormalize
        self.lambdaRate = lambdaRate
        self.mapping = mapping


    
    #Examples of feature values (A matrix of nxm) (dx(grad) is a matrix of 1xn)
    x = None

    #Examples of labeled values (A matrix of m x labelCount)
    y = None

    #Number of labeled values (#columns of y)
    labelCount = 1

    #Classes in classification problems
    classes = [1.0]

    #Number of features
    n = 0

    #Number of examples
    m = 0.0
        
    #Mean of features
    mu = float(0)

    #Standard deviation of features
    sigma = float(0)

    #The regularization rate
    lambdaRate = float(0)

    #A flag indicating if x0 should be included
    includeBias = False

    #A flag indicating if data should be normalized before training
    doNormalize = False
           
    #A tuple of mapping feature indices and a degree to which features should be mapped (to polynomial set)
    mapping = 0

    mapping_max_size_mb=None

    is_initialized = False

    def load_data(self, input_data, delimiter=",", x_range=None, labeled_range=None, verbose=True):
        data = None

        if isinstance(input_data, str):
            filepath = input_data
            with open(filepath, 'rb') as dataFile:
                data = np.loadtxt(dataFile,delimiter=delimiter)
        elif isinstance(input_data, types.GeneratorType):
            gen = input_data
            data = list(gen)
        elif isinstance(input_data, np.matrixlib.defmatrix.matrix):
            data = input_data
        elif isinstance(input_data, list) or isinstance(input_data, np.ndarray):
            data = np.asarray(input_data, dtype=np.float64)
        else:
            raise TypeError("Argument \"file_or_generator\" must be either a valid file path, list, array, matrix or a generator object.")

        columnCnt = data.shape[1]

        if columnCnt < 2:
            raise ValueError("The data in the file was not acceptable. A matrix of 2 or more columns is expected.")

        labeled_range = (columnCnt - self.labelCount, columnCnt) if labeled_range is None else labeled_range
        x_range = (0, columnCnt - self.labelCount) if x_range is None else x_range

        self.y = data[:, labeled_range[0] : labeled_range[1]]
        self.x = data[:, x_range[0] : x_range[1]].transpose()

        self.n = self.x.shape[0]
        self.m = self.x.shape[1]

        if self.doNormalize:
            self.normalizeFeatures()

        #Important: It should happen before including the bias value
        self.x = self.map_features(verbose)
        self.n = self.x.shape[0]

        if not self.is_initialized:
            self.initialize_properties()

    def map_features(self, verbose=True):
        if self.mapping:
            mapped = np.ones([1, self.m], dtype = np.float)

            mapping_degree = self.mapping[1]
            mapping_features = self.mapping[0]

            if verbose:
                print("Mapping features " + str(mapping_features) + " by degree " + str(mapping_degree))
                print("Current number of features: %d"%self.x.shape[0])
                print("Current size of features on memory: %dMB"%(math.ceil(sys.getsizeof(self.x)/(1024*1024))))

            #Will iterate through features, in each iteration a feature is picked, extended to polynominal set and cross multiplied by the multiplications from previous iteration (starting from 1)
            for featureIndex in range(self.n):
                feature = np.matrix(self.x[featureIndex, :])
                if featureIndex in mapping_features:
                    featurePol = mh.get_polynomial(feature, mapping_degree)
                    mapped = mh.cross_multiplication(mapped, featurePol, axis=0)
                
                    if self.mapping_max_size_mb and sys.getsizeof(mapped)/(1024*1024) > self.mapping_max_size_mb:
                        raise StopIteration("Size exceeded %gMB"%self.mapping_max_size_mb)
            
            for featureIndex in range(self.n):
                if not featureIndex in mapping_features:
                    feature = np.matrix(self.x[featureIndex, :])
                    mapped = np.append(mapped,feature,axis=0)
            
            if verbose:
                print("Size of features on memory: %dMB"%(math.ceil(sys.getsizeof(mapped)/(1024*1024))))
                print("New number of features: %d"%mapped.shape[0])
            return mapped

        return self.x

    def isEmpty(self):
        return self.x is None or self.y is None

    def normalizeFeatures(self, axis=1, ddof=1):
        self.x, self.mu, self.sigma = mh.normalizeMatrix(self.x, axis, ddof)

    def regularize_cost(self, cost, theta):
        if (self.lambdaRate <= 0.0):
            return cost

        #Exclude default factor
        if (self.includeBias):
            theta[0] = 0.0

        cost = cost + (1.0 / (2.0 * self.m)) * self.lambdaRate * np.sum(np.power(theta, 2))
        return cost

    def regularize_grads(self, grads, theta):
        if (self.lambdaRate <= 0.0):
            return grads

        #Exclude default factor
        if (self.includeBias):
            theta[0] = 0.0

        grads = grads + (1.0 / (self.m)) * self.lambdaRate * theta
        return grads

    def calc_training_accuracy(self):
        p = self.predict()
        return (p == self.y).mean() * 100

    @abstractmethod
    def train(self, to_convergence=True): pass

    @abstractmethod
    def initialize_properties(self):
        self.is_initialized = True

    @abstractmethod
    def compute_cost(self, theta, classIndex): pass

    @cache_result(maxsize=None)
    def get_labeled_set(self, classIndex=0):
        #Class index is ignored, it will be used in classification algorithms by overriding this method
        return np.matrix(self.y[:, 0]).transpose()

    @abstractmethod    
    def train(self): pass

    @abstractmethod
    def predict(self):pass

    @abstractmethod
    def save_snapshot(self): pass

    @abstractmethod
    def restore_snapshot(self): pass