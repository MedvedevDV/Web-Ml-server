from pathlib import Path
from abc import ABC, abstractmethod
import numpy as np
import pickle

class Model(ABC):
    """Базовый итерфейс модели"""
    def __init__(self, **params):
        self.model = None
        self.params = params

    @abstractmethod
    def fit(self, data: np.ndarray, tagets: np.ndarray):
        """Обучить модель на данных data с метками tagets."""
        pass

    @abstractmethod
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Вернуть предсказания для данных data."""
        pass

    # def metrics(self,)
    @classmethod
    def load(cls, path: Path) -> 'Model':
        """Загрузить модель с диска."""
        with open(path, 'rb') as f:
            return pickle.load(f)

    def upload(self, path: Path):
        """Сохраненить модели на диск."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def remove(self, path: Path):
        """Удалить обученную модель с диска."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod        
    def remove_all(path: Path) -> int:
        """Удалить все модели с диска и ворнуть количество удаленных."""
        count = 0
        for model_file in path.glob("*.pkl"):
            model_file.unlink()
            count += 1
        return count