data_path = "sample_data/data.txt";
classification_data_path = "sample_data/classification_data.csv";

def gradientdescent_load_data():
    gd = LinearRegression(1500, 0.01)
    basepath = os.path.dirname(__file__)
    filepath = os.path.abspath(os.path.join(basepath, data_path))
    gd.load_data(filepath)

    if gd.isEmpty():
        raise AssertionError("Loaded data was returned as null.")

def gradientdescent_cost():
    gd = LinearRegression(1500, 0.01)
    basepath = os.path.dirname(__file__)
    filepath = os.path.abspath(os.path.join(basepath, data_path))
    gd.load_data(filepath)
    theta = gd.theta[0,:]
    cost = gd.compute_cost(theta, 0)
    print("Cost is: " + cost)
   

def gradientdescent_cost_minimization():
    #Scenario 1: Gradient descent with 1 feature with default feature, no normalization
    gd = LinearRegression(1500, 0.01, True)
    basepath = os.path.dirname(__file__)
    filepath = os.path.abspath(os.path.join(basepath, data_path))
    gd.load_data(filepath)
    gd.train()    

    #Scenario 2: Gradient descent with 2 features with default feature and normalization
    gd = LinearRegression(100, 0.5, True, True)
    basepath = os.path.dirname(__file__)
    filepath = os.path.abspath(os.path.join(basepath, data_path))
    gd.load_data(filepath)
    gd.train()
    
def logistic_gradientdescent_cost_minimization():
    gd = LogisticRegression(400, 0.01, True)
    basepath = os.path.dirname(__file__)
    filepath = os.path.abspath(os.path.join(basepath, data_path))
    gd.load_data(filepath)
    gd.train(CostMinimizationAlgorithms.fmin)
    gd.plot_learning_curve()
    accuracy = gd.calc_training_accuracy()
    print("Accuracy: " + accuracy)

    #TODO: Values for fmin are a bit off, it was checked with the sample code, it was giving the same value, most probabely it is the implementation difference in fmin
    #TODO: Normal gradient is descending on cost 0.6 while fmin is giving 0.2, why it is so high in normal way?!!

    
def onevsall_cost_minimization():
    classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    gd = VectorizedLogisticRegression(50, 0.0, classes, includeBias=True, lambdaRate=0.1)
    basepath = os.path.dirname(__file__)
    filepath = os.path.abspath(os.path.join(basepath, classification_data_path))
    gd.load_data(filepath)
    gd.train(CostMinimizationAlgorithms.fmin)

    gd.plot_learning_curve()

def gradientdescent_learningCurve():
    gd = LinearRegression(1500, 0.01, True)
    basepath = os.path.dirname(__file__)
    filepath = os.path.abspath(os.path.join(basepath, data_path))
    gd.load_data(filepath)
    gd.train()
    gd.plot_learning_curve()

    gd = LinearRegression(100, 0.5, True, True)
    basepath = os.path.dirname(__file__)
    filepath = os.path.abspath(os.path.join(basepath, data_path))
    gd.load_data(filepath)
    gd.train()
    gd.plot_learning_curve()

def gradientdescent_featureNormalization():
    gd = LinearRegression(1500, 0.01)
    basepath = os.path.dirname(__file__)
    filepath = os.path.abspath(os.path.join(basepath, data_path))
    gd.load_data(filepath)
    gd.normalizeFeatures()
    print("mu: " + str(gd.mu))
    print("sigma: " + str(gd.sigma))
                  
def feature_mapping():      
    gd = LogisticRegression(1500, 0.01)
    basepath = os.path.dirname(__file__)
    filepath = os.path.abspath(os.path.join(basepath, data_path))
    gd.load_data(filepath)
    gd.mappingDegree = 6
    result = gd.map_features()
    #TODO:Finish