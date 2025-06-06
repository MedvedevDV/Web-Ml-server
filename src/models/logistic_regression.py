import numpy as np
from sklearn.linear_model import LogisticRegression
from .model import Model

class LogisticRegressionModel(Model):
    def __init__(self, **params):
        self.params = params

    def fit(self, data: np.ndarray, targets: np.ndarray):
        self.model = LogisticRegression(**self.params)
        self.model.fit(data, targets)
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        return self.model.predict(data)