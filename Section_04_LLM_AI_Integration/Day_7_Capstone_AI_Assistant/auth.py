"""JWT auth — demo user table for the capstone."""

import os
from datetime import datetime, timedelta

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
ALGO = "HS256"
TOKEN_TTL = timedelta(hours=8)

pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")

_USERS = {
    "alice": pwd.hash("wonderland"),
    "bob":   pwd.hash("builder"),
}

oauth2 = OAuth2PasswordBearer(tokenUrl="/auth/token")


def authenticate(username: str, password: str) -> bool:
    stored = _USERS.get(username)
    return bool(stored) and pwd.verify(password, stored)


def issue_token(username: str) -> str:
    payload = {"sub": username, "exp": datetime.utcnow() + TOKEN_TTL}
    return jwt.encode(payload, SECRET, algorithm=ALGO)


def current_user(token: str = Depends(oauth2)) -> str:
    err = HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid token")
    try:
        payload = jwt.decode(token, SECRET, algorithms=[ALGO])
    except JWTError:
        raise err
    username = payload.get("sub")
    if not username:
        raise err
    return username
