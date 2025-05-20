from fastapi import FastAPI 
from .ml.schemas import QueryText
from .routers import router as tasks_router

app = FastAPI()

app.include_router(tasks_router)

@app.get("/") 
async def read_root(): 
    return {'message': 'Welcome to Manager API!'}

@app.post("/predict/") 
async def get_prediction(item: QueryText): 
    return {"text": item.text}