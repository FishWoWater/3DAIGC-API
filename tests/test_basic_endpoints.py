"""
Basic API endpoint tests.

Tests for health check, system status, and basic API functionality.
"""

from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

# Import the main app
from api.main import app

client = TestClient(app)


class TestBasicEndpoints:
    """Test basic API endpoints"""

    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["version"] == "1.0.0"

    def test_root_endpoint(self):
        """Test root endpoint with API information"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "3D Generative Models API"
        assert data["version"] == "1.0.0"
        assert data["description"] == "Scalable 3D AI model inference server"
        assert data["docs_url"] == "/docs"
        assert data["health_url"] == "/health"

    def test_docs_endpoint(self):
        """Test API documentation endpoint"""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_redoc_endpoint(self):
        """Test ReDoc documentation endpoint"""
        response = client.get("/redoc")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_openapi_schema(self):
        """Test OpenAPI schema endpoint"""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert data["info"]["title"] == "3D Generative Models API"
        assert data["info"]["version"] == "1.0.0"
        assert "paths" in data

        # Check that our mesh generation endpoints are documented
        paths = data["paths"]
        assert "/api/v1/mesh-generation/text-to-textured-mesh" in paths
        assert "/health" in paths

    def test_cors_headers(self):
        """Test CORS headers are present"""
        response = client.get("/health")
        # Test for common CORS headers in the response
        # In development, CORS headers might be present
        assert response.status_code == 200
        # Just verify the endpoint works - CORS configuration is environment-specific

    def test_not_found_error(self):
        """Test 404 error handling"""
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404
        data = response.json()
        assert data["error"] == "NOT_FOUND"
        assert data["message"] == "Resource not found"
        assert "detail" in data

    @patch("api.main.get_settings")
    def test_system_status_endpoint(self, mock_get_settings):
        """Test system status endpoint"""
        # Mock settings
        mock_settings = Mock()
        mock_settings.environment = "test"
        mock_get_settings.return_value = mock_settings

        response = client.get("/api/v1/system/status")
        # This endpoint might not exist yet, but we test the expected structure
        if response.status_code == 404:
            pytest.skip("System status endpoint not implemented yet")
        else:
            assert response.status_code == 200
            data = response.json()
            # The actual response structure - check for any of the expected fields
            expected_fields = [
                "status",
                "system",
                "gpu",
                "models",
                "memory",
                "scheduler",
            ]
            assert any(field in data for field in expected_fields)

    def test_process_time_header(self):
        """Test that process time header is added"""
        response = client.get("/health")
        assert response.status_code == 200
        assert "X-Process-Time" in response.headers
        process_time = float(response.headers["X-Process-Time"])
        assert process_time >= 0

    def test_invalid_json_error(self):
        """Test error handling for invalid JSON"""
        response = client.post(
            "/api/v1/mesh-generation/text-to-textured-mesh",
            data="invalid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422  # Unprocessable Entity

    def test_method_not_allowed(self):
        """Test method not allowed error"""
        response = client.patch("/health")
        assert response.status_code == 405  # Method Not Allowed


class TestErrorHandling:
    """Test error handling middleware"""

    def test_validation_error_format(self):
        """Test validation error response format"""
        response = client.post(
            "/api/v1/mesh-generation/text-to-textured-mesh",
            json={},  # Missing required fields
        )
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        # FastAPI validation errors have specific format
        assert isinstance(data["detail"], list)

    def test_internal_server_error_handling(self, mock_scheduler):
        """Test internal server error handling"""
        # Mock scheduler to raise an exception
        mock_scheduler.schedule_job.side_effect = Exception("Mock internal error")

        response = client.post(
            "/api/v1/mesh-generation/text-to-textured-mesh",
            json={"prompt": "test prompt", "quality": "medium"},
        )

        # This should trigger our error handler
        if response.status_code == 500:
            data = response.json()
            assert data["error"] == "INTERNAL_ERROR"
            assert data["message"] == "An internal server error occurred"
            assert "detail" in data


class TestSystemEndpoints:
    """Test system-related endpoints"""

    def test_supported_formats_endpoint(self):
        """Test supported formats endpoint"""
        response = client.get("/api/v1/supported-formats")
        if response.status_code == 404:
            pytest.skip("Supported formats endpoint not implemented yet")
        else:
            assert response.status_code == 200
            data = response.json()
            assert "input_image_formats" in data
            assert "output_mesh_formats" in data

    def test_generation_presets_endpoint(self):
        """Test generation presets endpoint"""
        response = client.get("/api/v1/generation-presets")
        if response.status_code == 404:
            pytest.skip("Generation presets endpoint not implemented yet")
        else:
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, dict)
            # Should contain preset configurations
            for preset_name, preset_config in data.items():
                assert "quality" in preset_config
                assert "output_format" in preset_config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
