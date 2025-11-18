
class Model:

    def __init__(self):
        # init your model here
        pass

    def train(self, params, X_train, y_train):
        """
        Description of the function. 
        
        :param params: ......
        :param X_train: ......
        :param y_train: ......
        :return: ......
        """
        # Your implementation goes here
        # For our case, this function should train the initialised model and return the train f1 score
        # Return a evaluation metric (f1 in this case) as a single float so the caller can make use of it
        pass

    def evaluate(self, X_test, y_test):
        """
        Description of the function. 
        
        :param X_test: ......
        :param y_test: ......
        :return: ......
        """
        # This function should use the trained model to predict the target for the test data and return the test f1 score        
        pass

    def get_default_params(self):
        """
        Description of the function. 
        
        :return: ......
        """
        # This function should return the default parameters to be used for training the model from scratch
        # The output of this function should be the same as the params parameter in the train function
        # Edit this function to return the optimized parameters for your model
        pass
