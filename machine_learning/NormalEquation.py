__author__ = "Mohammad Dabiri"
__copyright__ = "Free to use, copy and modify"
__credits__ = ["Mohammad Dabiri"]
__license__ = "MIT Licence"
__version__ = "0.0.1"
__maintainer__ = "Mohammad Dabiri"
__email__ = "moddabiri@yahoo.com"

from machine_learning.LearnerBase import LearnerBase
import numpy as np

class NormalEquation(LearnerBase):
    def __init__(self, includeBias = False, doNormalize = False):
        return super().__init__(includeBias, doNormalize)

    def train(self, y):
        xTrans = self.x.transpose()
        self.theta = np.matrix(np.dot(np.dot(np.linalg.inv(np.dot(self.x, xTrans)), self.x), y)).transpose()

    def initialize_properties(self):
        no_of_classes = len(self.classes)

        if self.includeBias:
            self.theta = np.tile(np.zeros((self.labelCount, self.n + 1), dtype=np.float), (no_of_classes, 1))
            self.x = np.vstack((np.ones((1, self.m), dtype=np.float), self.x))
            self.n = self.n + 1
        else:
            self.theta = np.tile(np.zeros((self.labelCount, self.n), dtype=np.float), (no_of_classes, 1))

def main():
    ne = NormalEquation(includeBias= True, doNormalize = False)
    basepath = os.path.dirname(__file__)
    filepath = os.path.abspath(os.path.join(basepath, "Sample_Data/ex1data2.txt"))
    ne.load_data(filepath)
    ne.train(ne.y)

if __name__ == "__main__":
    main()