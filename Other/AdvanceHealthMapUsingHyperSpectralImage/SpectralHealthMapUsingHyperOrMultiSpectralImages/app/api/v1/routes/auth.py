"""
Authentication routes
"""
from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta
from app.core.security import SecurityManager
from app.core.config import settings
from app.schemas.user import UserLogin, Token, UserCreate
from app.services.user_service import UserService
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Login endpoint to get access token
    """
    try:
        # Authenticate user
        user = await UserService.authenticate_user(form_data.username, form_data.password)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create access token
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = SecurityManager.create_access_token(
            data={"sub": str(user["id"]), "roles": user.get("roles", ["user"])},
            expires_delta=access_token_expires
        )
        
        logger.info(f"User {user['username']} logged in successfully")
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        }
        
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/register", response_model=dict)
async def register(user_data: UserCreate):
    """
    Register new user
    """
    try:
        # Check if user already exists
        existing_user = await UserService.get_user_by_email(user_data.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create new user
        new_user = await UserService.create_user(user_data)
        
        logger.info(f"New user registered: {user_data.email}")
        
        return {
            "message": "User registered successfully",
            "user_id": new_user["id"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/refresh", response_model=Token)
async def refresh_token(current_user: dict = Depends(SecurityManager.verify_token)):
    """
    Refresh access token
    """
    try:
        # Create new access token
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = SecurityManager.create_access_token(
            data={"sub": current_user["sub"], "roles": current_user.get("roles", ["user"])},
            expires_delta=access_token_expires
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        }
        
    except Exception as e:
        logger.error(f"Token refresh error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )