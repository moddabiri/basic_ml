from enum import Enum

class CostMinimizationAlgorithms(Enum):
    gradient = 0
    fmin = 1
    fmin_unc = 2
    fmin_cg = 3         #It is more efficient when training data is large
    