from pydantic import BaseModel 
 
class QueryText(BaseModel): 
    text: str