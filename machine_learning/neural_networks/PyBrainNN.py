__author__ = "Mohammad Dabiri"
__copyright__ = "Free to use, copy and modify"
__credits__ = ["Mohammad Dabiri"]
__license__ = "MIT Licence"
__version__ = "0.0.1"
__maintainer__ = "Mohammad Dabiri"
__email__ = "moddabiri@yahoo.com"

from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import LinearLayer

import random

class PyBrainNN(object):
    """description of class"""
    def __init__(self, nn_structure, iterations, alpha, doNormalize = False, lambdaRate = 0.0, mappingDegree = 0, biasValue = 1):
        pass


    _m = 0
    _n = 0
    _batch_size = 0
    _iterations = 0
    _alpha = 0.0


    def get_batch_gen(self):
        #TODO: implement
        pass

    def train(self):
        # build the network
        fnn = buildNetwork(nn_structure, outclass=LinearLayer, bias=True)
        trainer = BackpropTrainer(fnn, momentum=0.1, verbose=True, weightdecay=0.01, learningrate=self._alpha, batchlearning=True)
        # repeat the batch training several times
        for i in xrange(200):
            # get a random order for the training examples for batch gradient descent
            random.shuffle(all_inds)
            # split the indexes into lists with the indices for each batch
            batch_inds = [all_inds[i:i+10] for i in xrange(0, len(all_inds), 10)]

            # train on each batch
            for inds in batch_inds:
                # rebuild the dataset
                ds = SupervisedDataSet(4, nb_classes=3)
                for x_i, y_i in zip(X[inds, :], y[inds]):
                    ds.appendLinked(x_i, y_i)
                ds._convertToOneOfMany()
                # train on the current batch
                trainer.trainOnDataset(ds)

        # make a dataset with all the iris data
        ds_all = ClassificationDataSet(4, nb_classes=3)
        for x_i, y_i in zip(X, y):
            ds_all.appendLinked(x_i, y_i)
        ds_all._convertToOneOfMany()

        # test the result
        # Note that we are testing on our training data, which is bad practice,
        # but it does demonstrate the network is trained
        print (sum(fnn.activateOnDataset(ds_all).argmax(axis=1) == y)/float(len(y)))


