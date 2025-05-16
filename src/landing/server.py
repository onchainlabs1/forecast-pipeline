from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import webbrowser
import subprocess
import threading
import time
import jwt
from datetime import datetime, timedelta

# Server configuration
PORT = 8002
STREAMLIT_PORT = 8501
MLFLOW_PORT = 8888
DIRECTORY = os.path.dirname(os.path.abspath(__file__))
SECRET_KEY = "your-secret-key-here"
ALGORITHM = "HS256"

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory=os.path.join(DIRECTORY, "static")), name="static")

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=60)  # Extend token lifetime
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

@app.get("/")
async def root():
    with open(os.path.join(DIRECTORY, "index.html"), "r") as file:
        html_content = file.read()
    return HTMLResponse(html_content)

@app.post("/token")
async def login(username: str = Form(...), password: str = Form(...)):
    # Validate credentials
    if (username == "admin" and password == "admin") or (username == "johndoe" and password == "secret"):
        # Create token with additional scopes
        token_data = {
            "sub": username,
            "scopes": ["predictions:read", "predictions:write", "admin"] if username == "admin" else ["predictions:read"]
        }
        access_token = create_access_token(token_data)
        return {"access_token": access_token, "token_type": "bearer", "username": username}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/users/me")
async def get_current_user(request: Request):
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            raise HTTPException(status_code=401, detail="Not authenticated")
        
        token = auth_header.split(' ')[1]
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
            
        return {"username": username}
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT) 