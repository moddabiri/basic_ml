__author__ = "Mohammad Dabiri"
__copyright__ = "Free to use, copy and modify"
__credits__ = ["Mohammad Dabiri"]
__license__ = "MIT Licence"
__version__ = "0.0.1"
__maintainer__ = "Mohammad Dabiri"
__email__ = "moddabiri@yahoo.com"

from enum import Enum

class CostMinimizationAlgorithms(Enum):
    gradient = 0
    fmin = 1
    fmin_unc = 2
    fmin_cg = 3         #It is more efficient when training data is large
    