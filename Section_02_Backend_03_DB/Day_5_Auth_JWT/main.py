"""
Day 5 - Auth: API Keys & JWT
----------------------------
A FastAPI app demonstrating password hashing with bcrypt and JWT-based auth.

Run:
    uvicorn main:app --reload

Try:
    # Login (form-encoded, per OAuth2 spec)
    curl -X POST http://127.0.0.1:8000/login \
         -d "username=alice&password=secret"

    # Use the token
    curl http://127.0.0.1:8000/me \
         -H "Authorization: Bearer <paste-token-here>"
"""

from datetime import datetime, timedelta, timezone

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext

# In production: load from env vars / secret manager. Never commit secrets.
SECRET_KEY = "change-me-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 15

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# In-memory user "DB"
users: dict[str, dict] = {
    "alice": {"username": "alice", "hashed_password": pwd_context.hash("secret")},
}

app = FastAPI(title="Day 5 - Auth")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """Build a signed JWT with an exp claim."""
    payload = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    payload["exp"] = expire
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    """Decode the JWT and return the username (sub)."""
    creds_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
    except JWTError:
        raise creds_error
    if username is None or username not in users:
        raise creds_error
    return username


@app.get("/")
def root():
    return {"message": "Day 5 - Auth (API keys & JWT)"}


@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = users.get(form_data.username)
    if not user or not pwd_context.verify(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="incorrect username or password",
        )
    token = create_access_token({"sub": user["username"]})
    return {"access_token": token, "token_type": "bearer"}


@app.get("/me")
def me(current_user: str = Depends(get_current_user)):
    return {"username": current_user}


if __name__ == "__main__":
    uvicorn.run(app)
