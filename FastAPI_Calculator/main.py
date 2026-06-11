from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Simple Calculator API", version="1.0.0")


class CalcRequest(BaseModel):
    a: float
    b: float


@app.get("/")
def root():
    return {"message": "Simple Calculator API", "endpoints": ["/add", "/subtract", "/multiply", "/divide"]}


@app.post("/add")
def add(req: CalcRequest):
    return {"operation": "addition", "a": req.a, "b": req.b, "result": req.a + req.b}


@app.post("/subtract")
def subtract(req: CalcRequest):
    return {"operation": "subtraction", "a": req.a, "b": req.b, "result": req.a - req.b}


@app.post("/multiply")
def multiply(req: CalcRequest):
    return {"operation": "multiplication", "a": req.a, "b": req.b, "result": req.a * req.b}


@app.post("/divide")
def divide(req: CalcRequest):
    if req.b == 0:
        raise HTTPException(status_code=400, detail="Division by zero is not allowed")
    return {"operation": "division", "a": req.a, "b": req.b, "result": req.a / req.b}


@app.get("/calculate")
def calculate(a: float, b: float, operation: str):
    """Query param based: /calculate?a=10&b=5&operation=add"""
    ops = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else None,
    }
    if operation not in ops:
        raise HTTPException(status_code=400, detail=f"Unknown operation '{operation}'. Use: add, subtract, multiply, divide")
    if operation == "divide" and b == 0:
        raise HTTPException(status_code=400, detail="Division by zero is not allowed")
    return {"operation": operation, "a": a, "b": b, "result": ops[operation]}
