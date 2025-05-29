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
    prefix='/ML_FuncTools',  # автоматический префикс для всех путей
    tags=['ML_FuncTools'],  # тег для группировки в документации
    responses={404: {'description': 'Not found'}}  # стандартные ответы
)


#===== Settings  =====
settings = Settings()

active_cpu_processes = 0
max_cpu_processes = max(1, settings.cpu_cores - 1)

training_models = {}
max_training_models = settings.max_loaded_models

loaded_models = {}
max_loaded_models = 1

process_exec = ProcessPoolExecutor(max_workers=max_cpu_processes)
process_futures = {}

FUNC_TOOLS_PATCH = settings.base_dir / 'models'
MODEL = {
    'logistic_regression': LogisticRegressionModel,
    'random_forest': RandomForestModel,
    'decision_tree': DecisionTreeModel
}
LOAD_SAVE_PATCH = FUNC_TOOLS_PATCH / 'auto_save_models'

process_lock = Lock()

#===== Process management =====
def process_status_exs(model_name: str):
    if active_cpu_processes>=max_cpu_processes:
        raise HTTPException(400, {                    
                    'error': 'No available slots. Try again later',
                    'active_processes': active_cpu_processes,
                    'max_processes': max_cpu_processes,
                    'available': 0
                    })
    
    elif model_name in training_models:
        raise HTTPException(400, f'{model_name} is load')
    
    elif len(training_models) >= max_training_models or len(loaded_models) >= max_loaded_models:
        raise HTTPException(400, f'Limit of uploaded models has been reached')
    
    
def process_callback(future, model_name: str):
    global active_cpu_processes
    active_cpu_processes -= 1
    if model_name in process_futures:
        del process_futures[model_name]
    
    try:
        future.result() 
    except Exception as e:
        print(f'Error in model {model_name} training: {str(e)}')


#===== Generate data =====
def train_model(model_name: str, params):
    csv_files = FUNC_TOOLS_PATCH / 'load_files'
    X_train = np.genfromtxt(csv_files / 'X_train.csv', delimiter=',', skip_header=1)
    y_train = np.genfromtxt(csv_files / 'y_train.csv', delimiter=',', skip_header=1)

    model = MODEL[model_name](**params)
    model.fit(X_train, y_train)

    model_save = LOAD_SAVE_PATCH / f'{model_name}.pkl'
    with open(model_save,'wb') as f:
        pickle.dump(model,f)


def get_predict(model_name,model, **params):
    X_test = np.genfromtxt('X_test.csv', delimiter=',', names=True, dtype=None)

    pred = model.predict(X_test)

    return pred 


#===== Routers =====
@router_ML.get('/get_models', summary='Доступные модели')
async def get_models():
    '''
    Получить список доступных моделей
    '''
    return list(MODEL.keys())


@router_ML.put('/fit_{model_name}')
async def fit(model_name: str):
    '''
    Train the specified model.
    '''
    global active_cpu_processes
    process_status_exs(model_name)
    
    with process_lock:
        try:
            params = {} #{'n_jobs': 1}
            future = process_exec.submit(train_model, model_name=model_name, params=params)
            process_futures[model_name] = future
            active_cpu_processes += 1
            future.add_done_callback(partial(process_callback, model_name=model_name))
        except Exception as e:
            active_cpu_processes -= 1
            raise HTTPException(500, f'Training failed: {str(e)}')

    return {'model': model_name, 'status': 'training_started', 'slots_used': f'{active_cpu_processes}/{max_cpu_processes}'}


@router_ML.put('/predict_{model_name}')
def pedict_model(model_name: str):
    model = training_models[model_name]
    csv_file = FUNC_TOOLS_PATCH / 'load_files'
    X_train = np.genfromtxt(csv_file / 'X_test.csv', delimiter=',', skip_header=1)
    pred = model.predict(X_train)

    return {'predictions': pred.tolist()}

@router_ML.put('/load_{model_name}')
def loading_model_to_inference(model_name: str):

    global loaded_models
    process_status_exs(model_name)

    model = LOAD_SAVE_PATCH / f'{model_name}.pkl'
    if not model.exists():
        raise HTTPException(404, f'{model_name} model is not found')
    
    try:
        with open(model, 'rb') as f:
            loaded_models[model_name] = pickle.load(f)

        return {'status': 'Models loaded successfully', 'model_name': model_name}
    except Exception as e:
        active_cpu_processes -= 1
        return HTTPException(400, f'{str(e)}')


@router_ML.delete('/unload_{model_name}')
async def unload_model(model_name: str):
    '''Unloads the model from memory'''
    global training_models, process_futures
    
    if model_name not in training_models:
        raise HTTPException(404, f'{model_name} model is not loaded into memory')
    
    del training_models[model_name]
    if model_name in process_futures:
        process_futures[model_name].cancel()

    return {'status': 'success', 'message': f'{model_name} model has been unloaded'}


@router_ML.delete('/remove_{model_name}')
async def remove_madel(model_name: str):
    model_file = LOAD_SAVE_PATCH / f'{model_name}.pkl'

    if not model_file.exists():
        raise HTTPException(404, f'Model file {model_name}.pkl not found')
    
    model_file.unlink()
    return {'status': f'{model_name} model was remove'}


@router_ML.delete('/removeall')
async def remove_all_models():
    '''
    Delete all files
    '''
    try:
        files = LOAD_SAVE_PATCH.iterdir()
        # files_to_del = pt.joinpath(FUNC_TOOLS_PATCH,'auto_save_models').iterdir()
        # print(files_to_del)
        file_names = []
        for file in files:
            print(file)
            file_names.append(file.name)
            file.unlink()
        
        return {'status': 'Models deleted successfully', 'files name': file_names}

    except Exception as e:
        return {'status': f'Error deleting files: {str(e)}'}


@router_ML.get('/status')
async def get_status():
    '''Получение статуса сервера'''
    return {
        'cpu_cores': settings.cpu_cores,
        'max_worker_processes': max_cpu_processes,
        'active_processes': active_cpu_processes,
        'available_slots': max_cpu_processes - active_cpu_processes,
        'running_models': list(process_futures.keys()),
        'training_models': list(training_models.keys()),
        'loaded_models': list(loaded_models.keys()),
    }
