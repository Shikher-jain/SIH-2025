"""
Test configuration and fixtures
"""
import pytest
import asyncio
from typing import Generator
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.db.base import Base
from app.db.session import get_db
from app.core.config import settings

# Test database URL
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

# Create test engine
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False}
)

# Create test session
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency for testing"""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


# Override the dependency
app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def client() -> Generator:
    """Create test client"""
    # Create test database tables
    Base.metadata.create_all(bind=engine)
    
    with TestClient(app) as test_client:
        yield test_client
    
    # Drop test database tables
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db() -> Generator:
    """Create database session for testing"""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def test_user_data():
    """Test user data"""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "full_name": "Test User",
        "password": "testpassword123",
        "roles": ["user"]
    }


@pytest.fixture
def test_spectral_data():
    """Test spectral analysis data"""
    return {
        "field_id": "test_field_001",
        "analysis_type": "full",
        "include_predictions": True,
        "hyperspectral_data": [[0.5, 0.6, 0.7] * 10] * 20,  # Mock data
        "sensor_data": {
            "air_temperature": 25.5,
            "humidity": 65.0,
            "soil_moisture": 45.2
        }
    }