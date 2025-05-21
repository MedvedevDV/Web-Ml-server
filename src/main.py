from fastapi import FastAPI 
from .ml.schemas import QueryText
from .ML_test_routers import router_ML as ML_test_routers
from .FASTAPI_requests import routers_req as fa_rec

app = FastAPI()

app.include_router(ML_test_routers)
app.include_router(fa_rec)

@app.get("/") 
async def read_root(): 
    return {'message': 'Welcome to Manager API!'}

@app.post("/predict/") 
async def get_prediction(item: QueryText): 
    return {"text": item.text}