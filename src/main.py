from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import JSONResponse
from .routers.ML_FuncTools import router_ML
from .routers.ML_files import router_ML_files
from typing import Annotated
from pathlib import Path as pt

app = FastAPI()

app.include_router(router_ML)
app.include_router(router_ML_files)
# app.include_router(routers_req)

#===== Configurations for unload file =====
LOAD_DIR = pt(__file__).parent / "ml/models/load_files"

@app.get("/") 
async def read_root(): 
    return {'message': 'Welcome to Manager API!'}
