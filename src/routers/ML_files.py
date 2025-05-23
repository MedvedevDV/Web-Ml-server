from fastapi import APIRouter, HTTPException, Form, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path as pt
from typing import List

router_ML_files = APIRouter(
    prefix="/ML_files",  # автоматический префикс для всех путей
    tags=["ML_files"],  # тег для группировки в документации
    responses={404: {"description": "Not found"}}  # стандартные ответы
)

#===== Configurations for unload file =====
LOAD_DIR = pt(__file__).parent.parent / "ml/models/load_files"

@router_ML_files.get('/get_files', summary="Get all loaded files")
async def get_files():
    model_names = [f.name for f in LOAD_DIR.glob('*')]
    return {"files": model_names}

@router_ML_files.post("/load", summary="Load file")
async def create_file(file: UploadFile = File(...)):
    """
    Upload a file for training the models
    """
    try:
        file_location = pt.joinpath(LOAD_DIR, file.filename)
        with open(file_location, "wb") as f:
            f.write(await file.read())
        
        return JSONResponse(
            status_code=200,
            content={"message": "File loaded successfully", 
                     "file name": file.filename, 
                     "path": str(file_location)}
        )
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error loading file: {str(e)}"}
        )

@router_ML_files.post("/delete/{file_name}", summary="Delete file")
async def delet_file(file_name: str):
    """
    Delete file by name
    """
    try:
        file_location = pt.joinpath(LOAD_DIR, file_name)
        file_location.unlink()
        
        return JSONResponse(
            status_code=200,
            content={"message": "File deleted successfully", 
                     "file name": file_name, 
                     "path": str(file_location)}
        )
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error deleting file: {str(e)}"}
        )
    
@router_ML_files.post("/delete_all", summary="Delete all files")
async def delet_file():
    """
    Delete all files
    """
    try:
        files_to_del = pt.joinpath(LOAD_DIR).iterdir()
        file_names = []
        for file in files_to_del:
            file_names.append(file.name)
            file.unlink()
        
        return JSONResponse(
            status_code=200,
            content={"message": "Files deleted successfully", 
                     "files name": file_names}
        )
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error deleting files: {str(e)}"}
        )