"""
Módulo para autenticação e autorização com JWT.
"""

import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Union

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# Configurações de segurança
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-for-dev-only")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Modelos Pydantic
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None
    scopes: List[str] = []

class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None
    scopes: List[str] = []

class UserInDB(User):
    hashed_password: str

# Contexto para hash de senhas
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 com Bearer token
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token",
    scopes={
        "predictions:read": "Ler previsões",
        "predictions:write": "Criar previsões",
        "admin": "Acesso de administrador"
    }
)

# Funções de utilitário para autenticação
def verify_password(plain_password, hashed_password):
    """Verifica se a senha fornecida corresponde ao hash armazenado."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    """Gera um hash para a senha fornecida."""
    return pwd_context.hash(password)

# Banco de dados simulado de usuários - em produção, use um banco de dados real
fake_users_db = {
    "johndoe": {
        "username": "johndoe",
        "full_name": "John Doe",
        "email": "johndoe@example.com",
        "hashed_password": get_password_hash("secret"),
        "disabled": False,
        "scopes": ["predictions:read"]
    },
    "admin": {
        "username": "admin",
        "full_name": "Admin User",
        "email": "admin@example.com",
        "hashed_password": get_password_hash("admin"),
        "disabled": False,
        "scopes": ["predictions:read", "predictions:write", "admin"]
    }
}

def get_user(db, username: str):
    """Recupera um usuário do banco de dados."""
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)

def authenticate_user(db, username: str, password: str):
    """Autentica um usuário pelo username e senha."""
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None):
    """Cria um token JWT de acesso."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Dependência para rotas protegidas
async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Valida o token JWT e retorna o usuário atual."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_scopes = payload.get("scopes", [])
        token_data = TokenData(username=username, scopes=token_scopes)
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    """Verifica se o usuário atual está ativo."""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# Verificação de escopos (autorização)
def validate_scopes(required_scopes: List[str]):
    """Cria uma dependência que verifica se o usuário tem os escopos necessários."""
    async def has_scopes(current_user: User = Depends(get_current_active_user)):
        for scope in required_scopes:
            if scope not in current_user.scopes:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions. Required scope: {scope}",
                )
        return current_user
    return has_scopes 