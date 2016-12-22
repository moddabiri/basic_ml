
def check_none(**kwargs):
    for [argName, value] in kwargs.items():
        if (value is None):
            raise TypeError("Argument was None: Argument: " + str(argName))