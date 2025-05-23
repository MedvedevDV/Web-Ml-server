import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .model import Model

class RandomForestModel(Model):
    def fit(self, data: np.ndarray, targets: np.ndarray):
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(data, targets)
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        return self.model.predict(data)