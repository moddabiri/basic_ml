import numpy as np
import importlib
import os.path

if not importlib.find_loader('matplotlib') is None:
    import matplotlib.pyplot as plt
else:
    print("WARNING! matplotlicb package was not installed on the vm. Plotting functionalities will not work.")


if not importlib.find_loader('MiniBatchSet') is None:
    from MiniBatchSet import MiniBatchSet
else:
    print("WARNING! MiniBatchSet class was not accessible. Loading without MiniBatchSet. Note that neural netwroks will not be available.")

from scipy import optimize
from machine_learning.LearnerBase import LearnerBase
from abc import ABCMeta, abstractmethod
from machine_learning.CostMinimizationAlgorithms import CostMinimizationAlgorithms

class CostMinimizerBase(LearnerBase):
    """description of class"""

    __metaclass__ = ABCMeta

    def __init__(self, iterations, alpha, includeBias = False, doNormalize = False, lambdaRate = 0.0, mapping = None, labelCount=1):
        self.iterations = iterations
        self.alpha = alpha

        if self.costs is None:
            self.costs = [None]

        return super().__init__(includeBias, doNormalize, lambdaRate, mapping, labelCount)

    #Theta values (will be filled after model is trained) (A matrix of   NoOfClasses x n)
    theta = None

    #Number of iterations
    iterations = 0

    #The learning rate
    alpha = float(0.01)

    #Final cost
    costs = None  

    #For fmin, iteration number cannot be extracted from callback, this counter is private
    lastIteration = 0

    def initialize_properties(self):
        no_of_classes = len(self.classes)

        if self.includeBias:
            self.theta = np.tile(np.zeros((self.labelCount, self.n + 1), dtype=np.float), (no_of_classes, 1))
            self.x = np.vstack((np.ones((1, self.m), dtype=np.float), self.x))
            self.n = self.n + 1
        else:
            self.theta = np.tile(np.zeros((self.labelCount, self.n), dtype=np.float), (no_of_classes, 1))

        super().initialize_properties()
 
    def compute_cost_grads(self, theta, classIndex):
        y = self.get_labeled_set(classIndex)
        cost = self.compute_cost(theta, classIndex)
        grads = self.compute_grads(theta, classIndex)
        return cost, grads
    
    def compute_store_cost(self, theta, classIndex):
        y = self.get_labeled_set(classIndex)
        cost = self.compute_cost(theta, classIndex)
        
        if (self.lastIteration >= self.iterations):
            #This scenario happens in fmin (it may exceed the iteration maximum)
            self.costs[classIndex].append(cost)
        else:
            self.costs[classIndex][self.lastIteration] = cost

        self.lastIteration += 1
        percentage_done = int((self.lastIteration/self.iterations)*100)
        if percentage_done > 0 and percentage_done % 10 == 0:
            print("%d%% of iterations passed. Currently at iteration %d. Cost is %g."%(percentage_done, self.lastIteration, cost))

        return cost

    def train_stochastic(self, batch_list, x_range=None, labeled_range=None, algorithm=CostMinimizationAlgorithms.gradient):
        if batch_list is None or not isinstance(batch_list, MiniBatchSet) or len(batch_list) <= 0:
            raise ValueError("Parameter batch_list must be a non-empty MiniBatchSet.")

        print("Starting the mini-batching training process with {0} mini-batches.".format(len(batch_list)))
        
        #Shuffle the list
        batch_list.reset()

        for iteration in range(self.iterations):
            batch = next(batch_list)
            self.load_data(batch, x_range=x_range, labeled_range=labeled_range, verbose=False)
            self.train(algorithm, to_convergence=False)
            print(">--[Training] [EPOCH {0}] [COST={1}]".format(iteration+1, self.costs[0][self.lastIteration-1]))

            #TODO: Remove this!!
            theta_path = "/usr/local/projects/ml/data/theta_sto.txt"
            a = np.asarray(self.theta[0])
            np.savetxt(theta_path, a, delimiter=",")
            #print(">--[Saving]: mini-batch {0}.".format(index+1))
            #self.save_snapshot()

    def train(self, algorithm = CostMinimizationAlgorithms.gradient, to_convergence=True):
        classCount = len(self.classes)
        for i in range(classCount): 
            self.costs[i] = [0.0] * self.iterations
            self.lastIteration = 0
            self.minimize_cost(i, algorithm, to_convergence)

    def minimize_cost(self, class_index, algorithm = CostMinimizationAlgorithms.gradient, to_convergence=True):
        classCount = len(self.classes)

        if algorithm == CostMinimizationAlgorithms.gradient:    
            self.minimize_cost_grad(class_index, to_convergence)
        elif algorithm == CostMinimizationAlgorithms.fmin or \
             algorithm == CostMinimizationAlgorithms.fmin_cg or \
             algorithm == CostMinimizationAlgorithms.fmin_unc:
            self.minimize_cost_fmin(class_index, algorithm, to_convergence)
        else:
            raise NotImplementedError('Request algorithm ({0}) is not implemented for cost minimization.'.format(algorithm.name))

    def minimize_cost_grad(self, classIndex, to_convergence=True):
        y = self.get_labeled_set()
        theta = np.matrix(self.theta[classIndex, :])

        iterations = self.iterations if to_convergence else 1
        for iteration in range(iterations):
            differentials = self.compute_grads(theta, classIndex)
            theta = theta - self.alpha * (1.0 / float(self.m)) * differentials
            self.compute_store_cost(theta, classIndex)

        self.theta[classIndex, :] = theta


    def minimize_cost_fmin(self, classIndex, algorithm = CostMinimizationAlgorithms.fmin, to_convergence=True):
        iterations = self.iterations if to_convergence else 1

        theta = np.matrix(self.theta[classIndex, :])
        cost = 0.0
        options = {'full_output': True, 'maxiter': iterations}

        if (algorithm == CostMinimizationAlgorithms.fmin or algorithm == CostMinimizationAlgorithms.fmin_unc):
            theta, cost, _, _, _ =  optimize.fmin(lambda t: self.compute_store_cost(t, classIndex=classIndex), theta, **options)
        elif (algorithm == CostMinimizationAlgorithms.fmin_cg):
            theta =  optimize.fmin_cg(lambda t: self.compute_store_cost(t, classIndex=classIndex), theta, maxiter=iterations)
        else:
            raise NotImplementedError("Requested gradient descent algorithm is not supported or not implemented.")

        self.theta[classIndex, :] = theta

    def plot_learning_curve(self):
        for cost in self.costs:
            if (not cost is None):
                plt.plot(cost)
                plt.ylabel('Cost')
                plt.xlabel('Iterations')
                plt.show()


    @abstractmethod
    def compute_grads(self, theta, classIndex): pass