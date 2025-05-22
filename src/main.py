from fastapi import FastAPI, Form
from .ml.schemas import QueryText
from .ML_FuncTools import router_ML
from .FASTAPI_requests import routers_req

app = FastAPI()

app.include_router(router_ML)
app.include_router(routers_req)

@app.get("/") 
async def read_root(): 
    return {'message': 'Welcome to Manager API!'}

@app.post("/predict/") 
async def get_prediction(item: QueryText): 
    return {"text": item.text}

