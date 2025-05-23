import numpy as np
from sklearn.tree import DecisionTreeClassifier
from .model import Model

class DecisionTreeModel(Model):
    def fit(self, data: np.ndarray, targets: np.ndarray):
        self.model = DecisionTreeClassifier(**self.params)
        self.model.fit(data, targets)
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        return self.model.predict(data)