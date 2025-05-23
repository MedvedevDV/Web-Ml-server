from pathlib import Path
from abc import ABC, abstractmethod
import numpy as np
import pickle

class Model(ABC):
    """Base interfase"""
    def __init__(self, **params):
        self.model = None
        self.params = params

    @abstractmethod
    def fit(self, data: np.ndarray, tagets: np.ndarray):
        """Обучить модель на данных data с метками tagets.
        Train a model besed on data with tags
        """
        pass

    @abstractmethod
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Return predictions for data."""
        pass

    # def metrics(self,)
    @classmethod
    def load(cls, path: Path) -> 'Model':
        """Load the model from disk. ."""
        with open(path, 'rb') as f:
            return pickle.load(f)

    def upload(self, path: Path):
        """Save model to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def remove(self, path: Path):
        """Delete the trained model from disk."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod        
    def remove_all(path: Path) -> int:
        """Delete all models from disk and check the number of deleted ones."""
        count = 0
        for model_file in path.glob("*.pkl"):
            model_file.unlink()
            count += 1
        return count