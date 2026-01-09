"""
Tests for user routes
"""
import pytest
from fastapi.testclient import TestClient
from app.services.user_service import UserService


class TestUserRoutes:
    """Test user routes"""
    
    def test_register_user(self, client: TestClient, test_user_data):
        """Test user registration"""
        response = client.post("/api/v1/auth/register", json=test_user_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "user_id" in data
        assert data["message"] == "User registered successfully"
    
    def test_register_duplicate_email(self, client: TestClient, test_user_data):
        """Test registration with duplicate email"""
        # Register first user
        client.post("/api/v1/auth/register", json=test_user_data)
        
        # Try to register again with same email
        response = client.post("/api/v1/auth/register", json=test_user_data)
        
        assert response.status_code == 400
        assert "Email already registered" in response.json()["detail"]
    
    def test_login_success(self, client: TestClient, test_user_data):
        """Test successful login"""
        # Register user first
        client.post("/api/v1/auth/register", json=test_user_data)
        
        # Login
        login_data = {
            "username": test_user_data["username"],
            "password": test_user_data["password"]
        }
        response = client.post("/api/v1/auth/login", data=login_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
    
    def test_login_invalid_credentials(self, client: TestClient):
        """Test login with invalid credentials"""
        login_data = {
            "username": "nonexistent",
            "password": "wrongpassword"
        }
        response = client.post("/api/v1/auth/login", data=login_data)
        
        assert response.status_code == 401
        assert "Incorrect username or password" in response.json()["detail"]
    
    def test_get_current_user(self, client: TestClient, test_user_data):
        """Test getting current user info"""
        # Register and login
        client.post("/api/v1/auth/register", json=test_user_data)
        
        login_data = {
            "username": test_user_data["username"],
            "password": test_user_data["password"]
        }
        login_response = client.post("/api/v1/auth/login", data=login_data)
        token = login_response.json()["access_token"]
        
        # Get current user
        headers = {"Authorization": f"Bearer {token}"}
        response = client.get("/api/v1/users/me", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == test_user_data["username"]
        assert data["email"] == test_user_data["email"]
    
    def test_unauthorized_access(self, client: TestClient):
        """Test access without token"""
        response = client.get("/api/v1/users/me")
        
        assert response.status_code == 403  # No Authorization header