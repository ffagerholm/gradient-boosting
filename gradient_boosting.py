import numpy as np
from scipy.optimize import minimize
from sklearn.utils.validation import check_X_y
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeRegressor
from loss_functions import SquaredErrorLoss


class GradientBooster(BaseEstimator, ClassifierMixin):
    """Gradient Boosting model."""
    def __init__(self, n_iter=10, lr=0.1,
                 max_depth=2, loss_function=SquaredErrorLoss(),
                 verbose=False):
        assert 0. < lr <= 1.
        self.models = []
        self.gammas = []
        self.loss = loss_function
        self.n_iter = n_iter
        self.lr = lr
        self.max_depth = max_depth
        self.verbose = verbose
        self.bias = None
        # array for storing training loss
        self.loss_hist = np.zeros(n_iter)
    
    def fit(self, X, y):
        """Fit the model to training data"""
        # Validate training data
        X, y = check_X_y(X, y, y_numeric=True)
        # calculate base score
        self.bias = y.mean()
        h0 = np.ones_like(y)*self.bias
        H = self.lr * h0

        for i in range(self.n_iter):
            # Compute gradient of the loss-function for 
            # the current prediction
            grad = self.loss.gradient(y, H)
            # Fit new regression tree to negative gradient
            tree = DecisionTreeRegressor(max_depth=self.max_depth).fit(X, -grad)
            
            # predicted score from the new model
            h = tree.predict(X)

            # Compute weight factor gamma for new tree 
            L = lambda g: self.loss(y, H + g*h).mean()
            # find find gamma that minimizes the loss
            res = minimize(L, x0=0.0)
            g = res.x[0]
            
            # add new model and its weight
            self.models.append(tree)
            self.gammas.append(g)
            # Add new weighted prediction 
            # with regularization (learning rate)
            H = H + self.lr*g*h
            # Compute current mean loss on training set
            self.loss_hist[i] = self.loss(y, H).mean()

        return self
    
    def _predict_base(self, X):
        """Predict the base score"""
        h0 = np.ones(X.shape[0])*self.bias
        H = self.lr * h0
        for tree, g in zip(self.models, self.gammas):
            H += self.lr * g * tree.predict(X)
        return H
        
    def predict(self, X):
        """Compute base score and apply link function"""
        H = self._predict_base(X)
        return self.loss.link(H)