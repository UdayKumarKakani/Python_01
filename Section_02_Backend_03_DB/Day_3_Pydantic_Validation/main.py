"""
Day 3 - Pydantic v2: Request Validation
---------------------------------------
A small FastAPI app showing how Pydantic models validate request bodies,
including nested models and `response_model`.

Run:
    uvicorn main:app --reload

Try in another terminal:
    curl -X POST http://127.0.0.1:8000/items \
         -H "Content-Type: application/json" \
         -d '{"name":"Pen","price":2.5,"quantity":10,"category":{"name":"Stationery"}}'
"""

from uuid import uuid4
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Day 3 - Pydantic Validation")


# --- Nested model ---
class Category(BaseModel):
    name: str = Field(..., min_length=2, max_length=30)


# --- Main request/response model ---
class Item(BaseModel):
    id: str | None = None  # auto-filled by the server
    name: str = Field(..., min_length=3, max_length=50, description="Item name")
    price: float = Field(..., gt=0, description="Must be > 0")
    quantity: int = Field(..., ge=0, description="Must be >= 0")
    category: Category
    
class Item2(BaseModel):
    id: str | None = None  # auto-filled by the server
    name: int = Field(..., gt=0, description="Must be > 0")
    price: int = Field(..., gt=0, description="Must be > 0")
    quantity: str = Field(..., min_length=1, description="Must be >= 0")
    category: Category    


@app.get("/")
def root():
    return {"message": "Day 3 - Pydantic Validation"}


@app.post("/items", response_model=Item)
def create_item(item: Item):
    # Pydantic has already validated `item` for us by this point
    item.id = str(uuid4())
    return item

@app.post("/items1", response_model=Item)
def create1_item(item: Item2):
    # Pydantic has already validated `item` for us by this point
    item.id = str(uuid4())
    return item