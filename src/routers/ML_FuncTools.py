from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path as pt
from typing import List

# Создаем экземпляр роутера
router_ML = APIRouter(
    prefix="/ML_FuncTools",  # автоматический префикс для всех путей
    tags=["ML_FuncTools"],  # тег для группировки в документации
    responses={404: {"description": "Not found"}}  # стандартные ответы
)

#Получить все доступные модели
@router_ML.get('/get_models', summary="Доступные модели")
async def get_models():
    """
    Получить список доступных моделей
    """
    path = pt(__file__).parent.parent / "ml/models"
    return [
    f.stem for f in path.glob("*.py") 
    if f.name != "__init__.py" and f.name != "model.py"
    ]


# # Эндпоинт для создания новой задачи
# @router.post("/", response_model=Task)
# async def create_task(task: Task):
#     fake_tasks_db.append(task)
#     return task

# # Эндпоинт для получения конкретной задачи
# @router.get("/{task_id}", response_model=Task)
# async def read_task(task_id: int):
#     task = next((t for t in fake_tasks_db if t.id == task_id), None)
#     if task is None:
#         raise HTTPException(status_code=404, detail="Task not found")
#     return task