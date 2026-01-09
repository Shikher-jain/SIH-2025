"""
User schemas for request/response validation
"""
from typing import Optional, List
from pydantic import BaseModel, EmailStr, validator
from datetime import datetime


class UserBase(BaseModel):
    """Base user schema"""
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    is_active: bool = True


class UserCreate(UserBase):
    """Schema for creating new user"""
    password: str
    roles: List[str] = ["user"]
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v
    
    @validator('roles')
    def validate_roles(cls, v):
        allowed_roles = ['user', 'researcher', 'admin']
        for role in v:
            if role not in allowed_roles:
                raise ValueError(f'Invalid role: {role}')
        return v


class UserUpdate(BaseModel):
    """Schema for updating user"""
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    is_active: Optional[bool] = None


class UserLogin(BaseModel):
    """Schema for user login"""
    username: str
    password: str


class UserResponse(UserBase):
    """Schema for user response"""
    id: int
    roles: List[str]
    created_at: datetime
    updated_at: Optional[datetime] = None
    last_login: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class Token(BaseModel):
    """Token response schema"""
    access_token: str
    token_type: str
    expires_in: int


class TokenData(BaseModel):
    """Token data schema"""
    user_id: Optional[str] = None
    roles: List[str] = []