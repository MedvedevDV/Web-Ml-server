import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .model import Model

class RandomForestModel(Model):
    def __init__(self, **params):
        default_params = {
            "n_estimators": 50,
            "max_depth": 20,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt", 
            "bootstrap": True,
        }
        self.params = {**default_params, **params}
        
    def fit(self, data: np.ndarray, targets: np.ndarray):
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(data, targets)
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        return self.model.predict(data)