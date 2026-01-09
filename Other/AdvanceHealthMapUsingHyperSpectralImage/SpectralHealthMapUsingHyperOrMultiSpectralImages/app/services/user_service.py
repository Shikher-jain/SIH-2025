"""
User service for business logic
"""
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from app.models.user import User
from app.schemas.user import UserCreate, UserUpdate
from app.core.security import SecurityManager
from app.db.session import SessionLocal
from app.utils.logger import get_logger

logger = get_logger(__name__)


class UserService:
    """Service for user-related operations"""
    
    @staticmethod
    async def create_user(user_data: UserCreate) -> Dict[str, Any]:
        """Create a new user"""
        db = SessionLocal()
        try:
            # Hash password
            hashed_password = SecurityManager.get_password_hash(user_data.password)
            
            # Create user instance
            user = User(
                username=user_data.username,
                email=user_data.email,
                full_name=user_data.full_name,
                hashed_password=hashed_password,
                roles=user_data.roles,
                is_active=user_data.is_active
            )
            
            # Save to database
            db.add(user)
            db.commit()
            db.refresh(user)
            
            logger.info(f"Created user: {user.username}")
            
            return {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name,
                "roles": user.roles,
                "is_active": user.is_active,
                "created_at": user.created_at
            }
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating user: {str(e)}")
            raise
        finally:
            db.close()
    
    @staticmethod
    async def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user credentials"""
        db = SessionLocal()
        try:
            # Find user by username or email
            user = db.query(User).filter(
                (User.username == username) | (User.email == username)
            ).first()
            
            if not user:
                return None
            
            # Verify password
            if not SecurityManager.verify_password(password, user.hashed_password):
                return None
            
            # Update last login
            from sqlalchemy.sql import func
            user.last_login = func.now()
            db.commit()
            
            return {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name,
                "roles": user.roles,
                "is_active": user.is_active
            }
            
        except Exception as e:
            logger.error(f"Error authenticating user: {str(e)}")
            return None
        finally:
            db.close()
    
    @staticmethod
    async def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            
            if not user:
                return None
            
            return {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name,
                "roles": user.roles,
                "is_active": user.is_active,
                "is_superuser": user.is_superuser,
                "created_at": user.created_at,
                "updated_at": user.updated_at,
                "last_login": user.last_login
            }
            
        except Exception as e:
            logger.error(f"Error getting user by ID: {str(e)}")
            return None
        finally:
            db.close()
    
    @staticmethod
    async def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
        """Get user by email"""
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.email == email).first()
            
            if not user:
                return None
            
            return {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name,
                "roles": user.roles,
                "is_active": user.is_active
            }
            
        except Exception as e:
            logger.error(f"Error getting user by email: {str(e)}")
            return None
        finally:
            db.close()
    
    @staticmethod
    async def update_user(user_id: int, user_update: UserUpdate) -> Optional[Dict[str, Any]]:
        """Update user information"""
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            
            if not user:
                return None
            
            # Update fields if provided
            for field, value in user_update.dict(exclude_unset=True).items():
                setattr(user, field, value)
            
            db.commit()
            db.refresh(user)
            
            logger.info(f"Updated user: {user.username}")
            
            return {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name,
                "roles": user.roles,
                "is_active": user.is_active,
                "updated_at": user.updated_at
            }
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error updating user: {str(e)}")
            raise
        finally:
            db.close()
    
    @staticmethod
    async def list_users(skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """List all users with pagination"""
        db = SessionLocal()
        try:
            users = db.query(User).offset(skip).limit(limit).all()
            
            return [
                {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "full_name": user.full_name,
                    "roles": user.roles,
                    "is_active": user.is_active,
                    "created_at": user.created_at,
                    "last_login": user.last_login
                }
                for user in users
            ]
            
        except Exception as e:
            logger.error(f"Error listing users: {str(e)}")
            return []
        finally:
            db.close()
    
    @staticmethod
    async def delete_user(user_id: int) -> bool:
        """Delete user by ID"""
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            
            if not user:
                return False
            
            db.delete(user)
            db.commit()
            
            logger.info(f"Deleted user: {user.username}")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error deleting user: {str(e)}")
            return False
        finally:
            db.close()