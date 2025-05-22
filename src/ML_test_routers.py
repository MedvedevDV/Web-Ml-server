from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

# Создаем экземпляр роутера
router_ML = APIRouter(
    prefix="/ML_FuncTools_test",
    tags=["ML_FuncTools_test"],
    responses={404: {"description": "Not found"}} 
)

# Модель Pydantic для данных
class Task(BaseModel):
    id: int
    title: str
    description: str = None
    completed: bool = False

# "База данных" (временное хранилище)
fake_tasks_db = [
    Task(id=1, title="Buy groceries", description="Milk, eggs, bread"),
    Task(id=2, title="Learn FastAPI", completed=True)
]


# Эндпоинт для создания новой задачи
@router_ML.post("/", response_model=Task)
async def create_task(task: Task):
    fake_tasks_db.append(task)
    return task

# Эндпоинт для получения конкретной задачи
@router_ML.get("/{task_id}", response_model=Task)
async def read_task(task_id: int):
    task = next((t for t in fake_tasks_db if t.id == task_id), None)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task