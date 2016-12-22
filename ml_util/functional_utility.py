__author__ = "Mohammad Dabiri"
__copyright__ = "Free to use, copy and modify"
__credits__ = ["Mohammad Dabiri"]
__license__ = "MIT Licence"
__version__ = "0.0.1"
__maintainer__ = "Mohammad Dabiri"
__email__ = "moddabiri@yahoo.com"

def check_none(**kwargs):
    for [argName, value] in kwargs.items():
        if (value is None):
            raise TypeError("Argument was None: Argument: " + str(argName))