import numpy as np
from scipy.special import expit as sigmoid

class SquaredErrorLoss:
    """Squared error loss function.
    """
    def __call__(self, y_true, y_pred):
        return 0.5 * (y_true - y_pred)**2
    
    def link(self, y_true):
        """Identity link function.
        Included to be compatible with the classification loss
        """
        return y_true
    
    def gradient(self, y_true, y_pred):
        return y_pred - y_true
    

class AbsoluteLoss:
    """Absolute value loss for robust regression.
    """
    def __call__(self, y_true, y_pred):
        return abs(y_true - y_pred)
    
    def link(self, y_true):
        return y_true
    
    def gradient(self, y_true, y_pred):
        return np.sign(y_pred - y_true)
    

class HuberLoss:
    """Huber loss for robust regression.
    https://en.wikipedia.org/wiki/Huber_loss
    """
    def __init__(self, delta=1):
        self.delta = delta
    
    def __call__(self, y_true, y_pred):
        absolute = abs(y_true - y_pred)
        l1 = 0.5 * (y_true - y_pred)**2
        l2 = self.delta * absolute - 0.5*self.delta**2
        return np.where(absolute < self.delta, l1, l2)
    
    def link(self, y_true):
        return y_true
        
    def gradient(self, y_true, y_pred):
        absolute = abs(y_true - y_pred)
        d1 = (y_true - y_pred)
        d2 = np.sign(y_true - y_pred)
        return np.where(absolute < self.delta, d1, d2)
    

class BinomialDevianceLoss:
    """Binary Cross entropy loss function (on the logit scale).
    https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_loss_function_and_logistic_regression
    https://en.wikipedia.org/wiki/Logit
    """
    def __call__(self, y_true, y_logits):
        return np.log1p(np.exp(y_logits)) - y_true*y_logits

    def link(self, y_logits):
        """Transforms logits to probabilities."""
        return sigmoid(y_logits)

    def gradient(self, y_true, y_logits):
        return sigmoid(y_logits) - y_true
    

class HingeLoss:
    """Hinge loss for classification.
    https://en.wikipedia.org/wiki/Hinge_loss
    """
    def __call__(self, y_true, y_base):
        t = 2*y_true - 1 
        return np.maximum(0, 1 - t*y_base)
    
    def link(self, y_base):
        return (y_base > 0).astype(int)
    
    def gradient(self, y_true, y_base):
        t = 2*y_true - 1 
        return np.where(0 < 1 - t*y_base, -t, np.zeros_like(y_true))