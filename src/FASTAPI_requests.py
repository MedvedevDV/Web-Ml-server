from fastapi import APIRouter, Path, Query
from typing import List, Dict, Union, Optional

routers_req = APIRouter(
    prefix="/FAPI_ROUTERS",
    tags=["FAPI_ROUTERS"]
)

# --- GET ---
@routers_req.get("/{item_id}")
async def read_item(item_id: int = Path(..., title="The ID of the item to get"),
                   q: Optional[str] = Query(None, max_length=50)):
    """
    Получает элемент по его ID.

    - **item_id**: ID элемента (целое число).
    - **q**: Необязательный параметр запроса (строка).
    """
    return {"item_id": item_id, "q": q}

@routers_req.get("/")
async def read_items(skip: int = 0, limit: int = 10) -> List[Dict[str, Union[str, int]]]:
    """
    Получает список элементов с пагинацией.

    - **skip**: Сколько элементов пропустить (по умолчанию 0).
    - **limit**: Максимальное количество элементов (по умолчанию 10).
    """
    fake_items_db = [
        {"item_name": "Foo", "item_id": 1},
        {"item_name": "Bar", "item_id": 2},
        {"item_name": "Baz", "item_id": 3},
    ]
    return fake_items_db[skip : skip + limit]

# --- POST ---
@routers_req.post("/")
async def create_item(item: Dict[str, Union[str, int, float]]):
    """
    Создает новый элемент.

    Тело запроса должно быть словарем (JSON) с произвольными ключами (строки)
    и значениями, которые могут быть строками, целыми числами или числами с плавающей точкой.
    **Внимание:** Валидация структуры словаря очень ограничена без Pydantic!
    """
    return item

# --- PUT ---
@routers_req.put("/{item_id}")
async def update_item(item_id: int, item: Dict[str, Union[str, int, float]]):
    """
    Полностью обновляет элемент с заданным ID.

    - **item_id**: ID элемента (целое число).
    - **item**:  Словарь (JSON) с *полным* новым представлением элемента.
      **Внимание:** Валидация структуры словаря очень ограничена без Pydantic!
    """
    return {"item_id": item_id, "updated_item": item}

# --- PATCH ---
@routers_req.patch("/{item_id}")
async def partial_update_item(item_id: int, item: Dict[str, Union[str, int, float]]):
    """
    Частично обновляет элемент с заданным ID.

    - **item_id**: ID элемента (целое число).
    - **item**: Словарь (JSON) с полями, которые нужно обновить.
      **Внимание:** Валидация структуры структуры словаря очень ограничена без Pydantic!
    """
    return {"item_id": item_id, "partially_updated_item": item}

# --- DELETE ---
@routers_req.delete("/{item_id}")
async def delete_item(item_id: int):
    """
    Удаляет элемент с заданным ID.

    - **item_id**: ID элемента (целое число).
    """
    return {"message": f"Item {item_id} deleted"}

# --- OPTIONS ---
@routers_req.options("/{item_id}")
async def options_item(item_id: int):
    """
    Возвращает информацию о доступных методах для элемента.
    """
    return {"allowed_methods": ["GET", "PUT", "PATCH", "DELETE", "OPTIONS"]}

# --- HEAD ---
@routers_req.head("/{item_id}")
async def head_item(item_id: int):
    """
    Возвращает заголовки ответа для элемента (без тела).
    """
    return

# --- TRACE ---
@routers_req.trace("/{item_id}")
async def trace_item(item_id: int):
    """
    Трассировка запроса
    """
    return {"item_id": item_id}