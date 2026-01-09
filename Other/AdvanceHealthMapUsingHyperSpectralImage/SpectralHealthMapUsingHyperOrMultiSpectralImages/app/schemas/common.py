"""
Common schemas
"""
from pydantic import BaseModel
from typing import Optional


class HealthCheck(BaseModel):
    """Health check response schema"""
    status: str
    message: str
    version: str
    timestamp: str


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str
    message: str
    details: Optional[str] = None