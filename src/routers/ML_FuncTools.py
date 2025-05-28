import pandas as pd
import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path as pt
from typing import Dict, List, Optional
from threading import Lock
from ..config.settings import Settings
from ..models.logistic_regression import LogisticRegressionModel
from ..models.random_forest import RandomForestModel
from ..models.decision_tree import DecisionTreeModel
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pickle
from functools import partial

# Создаем экземпляр роутера
router_ML = APIRouter(
    prefix="/ML_FuncTools",  # автоматический префикс для всех путей
    tags=["ML_FuncTools"],  # тег для группировки в документации
    responses={404: {"description": "Not found"}}  # стандартные ответы
)


#===== Settings  =====
settings = Settings()
active_cpu_processes = 0
max_cpu_processes = max(1, settings.cpu_cores - 1)
loaded_models = {}
max_loaded_models = settings.max_loaded_models
process_exec = ProcessPoolExecutor(max_workers=max_cpu_processes)
process_futures = {}


#===== Generate data =====
def train_model(model_name, params):
    csv_files = FUNC_TOOLS_PATCH / 'load_files'
    X_train = np.genfromtxt(csv_files / 'X_train.csv', delimiter=',', skip_header=1)
    y_train = np.genfromtxt(csv_files / 'y_train.csv', delimiter=',', skip_header=1)

    model = MODEL[model_name](**params)
    model.fit(X_train, y_train)

    model_save = LOAD_SAVE_PATCH / f'{model_name}.pkl'
    with open(model_save,'wb') as f:
        pickle.dump(model,f)


def get_predict(model_name,**params):
    X_test = np.genfromtxt('X_test.csv', delimiter=',', names=True, dtype=None)

    model = MODEL[model_name](params)
    pred = model.predict(X_test)

    return pred 

FUNC_TOOLS_PATCH = settings.base_dir / "models"
MODEL = {
    'logistic_regression': LogisticRegressionModel,
    'random_forest': RandomForestModel,
    'decision_tree': DecisionTreeModel
}
LOAD_SAVE_PATCH = FUNC_TOOLS_PATCH / 'auto_save_models'

process_lock = Lock()
#===== Process management =====
def process_callback(future, model_name: str):
    global active_cpu_processes, CPU_CORES
    active_cpu_processes -= 1
    CPU_CORES += 1
    if model_name in process_futures:
        del process_futures[model_name]
    
    try:
        future.result() 
    except Exception as e:
        print(f"Error in model {model_name} training: {str(e)}")

#===== Routers =====
@router_ML.get('/get_models', summary="Доступные модели")
async def get_models():
    """
    Получить список доступных моделей
    """
    return list(MODEL.keys())


@router_ML.put('/fit_{model_name}')
async def fit(model_name: str):
    """
    Train the specified model.
    """
    global active_cpu_processes, CPU_CORES
    if  active_cpu_processes>=max_cpu_processes:
        raise HTTPException(400, {                    
                    "error": "No available slots. Try again later",
                    "active_processes": active_cpu_processes,
                    "max_processes": max_cpu_processes,
                    "available": 0
                    })
    if len(loaded_models) >= settings.max_loaded_models:
        raise HTTPException(429,f"Limit of uploaded models has been reached ({settings.max_loaded_models})")
    
    try:
        params = {'solver': 'lbfgs', 'n_jobs': 1}
        future = process_exec.submit(train_model, model_name=model_name, params=params)
        process_futures[model_name] = future
        CPU_CORES -= 1
        active_cpu_processes += 1
        future.add_done_callback(partial(process_callback, model_name=model_name))
    except Exception as e:
        active_cpu_processes -= 1
        raise HTTPException(500, f"Training failed: {str(e)}")

    return {"model": model_name, "status": "training_started", "slots_used": f"{active_cpu_processes}/{max_cpu_processes}"}


@router_ML.get("/status")
async def get_status():
    """Получение статуса сервера"""
    return {
        "cpu_cores": settings.cpu_cores,
        "max_worker_processes": max_cpu_processes,
        "active_tasks": active_cpu_processes,
        "available_slots": max_cpu_processes - active_cpu_processes,
        "running_tasks": list(process_futures.keys())
    }
# @router_ML.put('/predict_{mopdel_name}')
# async def pred(mopdel_name: str):