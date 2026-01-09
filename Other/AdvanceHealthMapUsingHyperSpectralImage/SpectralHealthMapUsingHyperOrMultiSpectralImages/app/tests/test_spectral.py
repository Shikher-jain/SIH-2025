"""
Tests for spectral analysis routes
"""
import pytest
from fastapi.testclient import TestClient
import json


class TestSpectralRoutes:
    """Test spectral analysis routes"""
    
    def get_auth_headers(self, client: TestClient, test_user_data):
        """Helper to get authentication headers"""
        # Register and login
        client.post("/api/v1/auth/register", json=test_user_data)
        
        login_data = {
            "username": test_user_data["username"],
            "password": test_user_data["password"]
        }
        login_response = client.post("/api/v1/auth/login", data=login_data)
        token = login_response.json()["access_token"]
        
        return {"Authorization": f"Bearer {token}"}
    
    def test_analyze_spectral_data(self, client: TestClient, test_user_data, test_spectral_data):
        """Test spectral data analysis"""
        headers = self.get_auth_headers(client, test_user_data)
        
        response = client.post(
            "/api/v1/spectral/analyze",
            json=test_spectral_data,
            headers=headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "analysis_id" in data
        assert "field_id" in data
        assert data["field_id"] == test_spectral_data["field_id"]
        assert "processing_status" in data
    
    def test_get_user_fields(self, client: TestClient, test_user_data, test_spectral_data):
        """Test getting user fields"""
        headers = self.get_auth_headers(client, test_user_data)
        
        # First analyze some data to create fields
        client.post(
            "/api/v1/spectral/analyze",
            json=test_spectral_data,
            headers=headers
        )
        
        # Get user fields
        response = client.get("/api/v1/spectral/fields", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_get_models_status(self, client: TestClient, test_user_data):
        """Test getting models status"""
        headers = self.get_auth_headers(client, test_user_data)
        
        response = client.get("/api/v1/spectral/models/status", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
    
    def test_unauthorized_spectral_access(self, client: TestClient, test_spectral_data):
        """Test unauthorized access to spectral endpoints"""
        response = client.post("/api/v1/spectral/analyze", json=test_spectral_data)
        
        assert response.status_code == 403  # No Authorization header
    
    def test_invalid_analysis_type(self, client: TestClient, test_user_data, test_spectral_data):
        """Test invalid analysis type"""
        headers = self.get_auth_headers(client, test_user_data)
        
        invalid_data = test_spectral_data.copy()
        invalid_data["analysis_type"] = "invalid_type"
        
        response = client.post(
            "/api/v1/spectral/analyze",
            json=invalid_data,
            headers=headers
        )
        
        assert response.status_code == 422  # Validation error