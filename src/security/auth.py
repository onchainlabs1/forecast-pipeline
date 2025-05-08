"""
Module for authentication and authorization with JWT.
"""

import os
import base64
import json
import hmac
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Union

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

# Security settings
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-for-dev-only")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Pydantic models
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

# Password hashing context
# Simple password hashing function
def get_password_hash(password: str) -> str:
    """Generates a hash for the provided password."""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies if the provided password matches the stored hash."""
    return get_password_hash(plain_password) == hashed_password

# OAuth2 with Bearer token
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token",
    scopes={
        "predictions:read": "Read predictions",
        "predictions:write": "Create predictions",
        "admin": "Admin access"
    }
)

# Simulated user database - in production, use a real database
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
    """Retrieves a user from the database."""
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)

def authenticate_user(db, username: str, password: str):
    """Authenticates a user by username and password."""
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

# Simple JWT implementation
def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None):
    """Creates a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire.timestamp()})
    
    # Convert payload to JSON
    payload = json.dumps(to_encode).encode()
    
    # Create a simple JWT token (header.payload.signature)
    header = base64.urlsafe_b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode()).decode().rstrip("=")
    payload_b64 = base64.urlsafe_b64encode(payload).decode().rstrip("=")
    
    # Create signature
    signature = hmac.new(
        SECRET_KEY.encode(),
        f"{header}.{payload_b64}".encode(),
        hashlib.sha256
    ).digest()
    signature_b64 = base64.urlsafe_b64encode(signature).decode().rstrip("=")
    
    # Return the JWT token
    return f"{header}.{payload_b64}.{signature_b64}"

# JWT token validation
def decode_jwt(token: str) -> Dict[str, Any]:
    """Decodes and validates a JWT token."""
    try:
        # Split the token into parts
        header_b64, payload_b64, signature_b64 = token.split(".")
        
        # Add padding
        def add_padding(s):
            return s + "=" * (4 - len(s) % 4)
        
        # Verify signature
        signature = base64.urlsafe_b64decode(add_padding(signature_b64))
        expected_signature = hmac.new(
            SECRET_KEY.encode(),
            f"{header_b64}.{payload_b64}".encode(),
            hashlib.sha256
        ).digest()
        
        if not hmac.compare_digest(signature, expected_signature):
            raise ValueError("Invalid signature")
        
        # Decode payload
        payload = json.loads(base64.urlsafe_b64decode(add_padding(payload_b64)))
        
        # Check expiration
        if "exp" in payload and payload["exp"] < datetime.utcnow().timestamp():
            raise ValueError("Token expired")
        
        return payload
    except Exception as e:
        raise ValueError(f"Invalid token: {str(e)}")

# Dependency for protected routes
async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Validates the JWT token and returns the current user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_jwt(token)
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_scopes = payload.get("scopes", [])
        token_data = TokenData(username=username, scopes=token_scopes)
    except Exception:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    """Checks if the current user is active."""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# Scope verification (authorization)
def validate_scopes(required_scopes: List[str]):
    """Creates a dependency that checks if the user has the necessary scopes."""
    async def has_scopes(current_user: User = Depends(get_current_active_user)):
        for scope in required_scopes:
            if scope not in current_user.scopes:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions. Required scope: {scope}",
                )
        return current_user
    return has_scopes 