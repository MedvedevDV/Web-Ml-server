import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path as pt
from typing import List
from ..models.logistic_regression import LogisticRegressionModel
from ..models.random_forest import RandomForestModel
from ..models.decision_tree import DecisionTreeModel

# Создаем экземпляр роутера
router_ML = APIRouter(
    prefix="/ML_FuncTools",  # автоматический префикс для всех путей
    tags=["ML_FuncTools"],  # тег для группировки в документации
    responses={404: {"description": "Not found"}}  # стандартные ответы
)

FUNC_TOOLS_PATCH = pt(__file__).parent.parent / "models"
MODEL = {
    'logistic_regression': LogisticRegressionModel,
    'random_forest': RandomForestModel,
    'decision_tree': DecisionTreeModel
}
FILES_PATCH = pt(__file__).parent.parent / "models/load_files"


async def get_data(patch=FILES_PATCH):
    data_csv = pt.joinpath(patch, 'dataset_game_of_thrones_test.csv')
    test_csv = pt.joinpath(patch, 'game_of_thrones_test.csv')

    df_data = pd.read_csv(data_csv)
    df_test = pd.read_csv(test_csv)

    X = df_data.drop(['isAlive'], axis=1)
    y = df_data['isAlive']

    return {
        'data_train': X.values,
        'targets_train': y.values,
        'data_test': df_test.values
    }


#Получить все доступные модели
@router_ML.get('/get_models', summary="Доступные модели")
async def get_models():
    """
    Получить список доступных моделей
    """
    return [
    f.stem for f in FUNC_TOOLS_PATCH.glob("*.py") 
    if f.name != "__init__.py" and f.name != "model.py"
    ]

@router_ML.put('/fit_{mopdel_name}')
async def fit(mopdel_name: str):
    """
    Train the specified model.
    """
    model = MODEL[mopdel_name]()
    data_dict = await get_data()

    model.fit(data=data_dict['data_train'], targets=data_dict['targets_train'])

@router_ML.put('/predict_{mopdel_name}')
async def predict(mopdel_name: str):
    """
    Returns predicted data from the specified model.
    """
    model = MODEL[mopdel_name]()
    data_dict = await get_data()

    pred = model.predict(data_dict['data_test'])

    return pred