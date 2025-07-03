"""
Test configuration and utilities.

Provides common test fixtures, mocks, and utilities for the test suite.
This includes automatic scheduler dependency mocking for all tests.
"""

import os
import shutil
import tempfile
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_scheduler

# Import the main app and dependencies
from api.main import app

# Test data and fixtures
TEST_PROMPTS = [
    "A red dragon with detailed scales",
    "A blue crystal castle with towers",
    "A green futuristic robot",
    "A wooden sailing ship",
    "A golden crown with jewels",
]

TEST_INVALID_PROMPTS = [
    "",
    " ",
    "x" * 10000,  # Too long
]

TEST_FORMATS = ["glb", "obj"]
TEST_QUALITIES = ["low", "medium", "high"]
TEST_RESOLUTIONS = [256, 512, 1024, 2048]


@pytest.fixture(scope="function")
def mock_scheduler():
    """Create a mock scheduler for testing"""
    scheduler = Mock()
    scheduler.schedule_job = AsyncMock(return_value="test-job-123")
    scheduler.get_job_status = AsyncMock(
        return_value={
            "job_id": "test-job-123",
            "status": "completed",
            "feature": "text_to_textured_mesh",
            "result": {
                "mesh_path": "/tmp/test_mesh.glb",
                "texture_path": "/tmp/test_texture.png",
                "output_format": "glb",
            },
            "error": None,
        }
    )
    scheduler.register_model = Mock()
    scheduler.start = AsyncMock()
    scheduler.stop = AsyncMock()
    scheduler.models = {}
    return scheduler


@pytest.fixture(scope="function", autouse=True)
def auto_mock_scheduler(mock_scheduler):
    """Automatically mock the scheduler dependency for all tests"""
    # Override the dependency with our mock scheduler
    app.dependency_overrides[get_scheduler] = lambda: mock_scheduler

    yield mock_scheduler

    # Clean up the dependency override after each test
    app.dependency_overrides.clear()
    # app.state.scheduler = get_scheduler()


@pytest.fixture(scope="function")
def test_client():
    """Create a test client with mocked dependencies"""
    return TestClient(app)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_trellis_adapter():
    """Create a mock TRELLIS adapter for testing"""
    from adapters.trellis_adapter import TrellisTextToTexturedMeshAdapter
    from core.models.base import ModelStatus

    adapter = Mock(spec=TrellisTextToTexturedMeshAdapter)
    adapter.model_id = "mock-trellis"
    adapter.feature_type = "text_to_textured_mesh"
    adapter.status = ModelStatus.LOADED
    adapter.vram_requirement = 4096
    adapter.processing_count = 0
    adapter.supported_output_formats = ["glb", "obj"]

    adapter.load = AsyncMock(return_value=True)
    adapter.unload = AsyncMock(return_value=True)
    adapter.process = AsyncMock(
        return_value={
            "mesh_path": "/tmp/mock_mesh.glb",
            "texture_path": "/tmp/mock_texture.png",
            "output_format": "glb",
            "processing_time": 120.5,
        }
    )
    adapter.get_supported_formats = Mock(
        return_value={"input": ["text"], "output": ["glb", "obj"]}
    )
    adapter.get_info = Mock(
        return_value={
            "model_id": "mock-trellis",
            "status": ModelStatus.LOADED.value,
            "vram_requirement": 4096,
            "processing_count": 0,
            "supported_formats": {"input": ["text"], "output": ["glb", "obj"]},
        }
    )

    return adapter


@pytest.fixture
def sample_job_request():
    """Create a sample job request for testing"""
    from core.scheduler.job_queue import JobRequest

    return JobRequest(
        feature="text_to_textured_mesh",
        inputs={
            "prompt": "A red dragon",
            "output_format": "glb",
            "texture_resolution": 1024,
            "quality": "medium",
        },
        priority=1,
    )


class TestDataGenerator:
    """Generate test data for various scenarios"""

    @staticmethod
    def generate_valid_request():
        """Generate a valid text-to-textured-mesh request"""
        return {
            "prompt": "A beautiful crystal castle",
            "texture_prompt": "Sparkling blue crystal texture",
            "quality": "medium",
            "output_format": "glb",
            "texture_resolution": 1024,
            "model_name": "trellis",
        }

    @staticmethod
    def generate_invalid_requests():
        """Generate various invalid requests for testing validation"""
        return [
            {},  # Empty request
            {"prompt": ""},  # Empty prompt
            {"prompt": "test", "quality": "invalid"},  # Invalid quality
            {"prompt": "test", "output_format": "invalid"},  # Invalid format
            {"prompt": "test", "texture_resolution": 128},  # Too low resolution
            {"prompt": "test", "texture_resolution": 8192},  # Too high resolution
        ]


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "api: marks tests as API tests")
    config.addinivalue_line("markers", "adapter: marks tests as adapter tests")
    config.addinivalue_line("markers", "slow: marks tests as slow running")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")


def assert_valid_mesh_result(result):
    """Assert that a mesh generation result is valid"""
    assert isinstance(result, dict)
    assert "mesh_path" in result
    assert "output_format" in result
    assert os.path.splitext(result["mesh_path"])[1][1:] == result["output_format"]

    if "texture_path" in result:
        assert result["texture_path"].endswith((".png", ".jpg", ".jpeg"))

    if "processing_time" in result:
        assert isinstance(result["processing_time"], (int, float))
        assert result["processing_time"] > 0


def assert_valid_job_status(status):
    """Assert that a job status response is valid"""
    assert isinstance(status, dict)
    assert "job_id" in status
    assert "status" in status
    assert "feature" in status
    assert status["status"] in ["queued", "processing", "completed", "error"]

    if status["status"] == "completed":
        assert status["result"] is not None
        assert_valid_mesh_result(status["result"])

    if status["status"] == "error":
        assert status["error"] is not None
        assert isinstance(status["error"], str)
        assert len(status["error"]) > 0


def assert_valid_api_response(response, expected_status=200):
    """Assert that an API response is valid"""
    assert response.status_code == expected_status

    if expected_status == 200:
        data = response.json()
        assert isinstance(data, dict)

    # Check for required headers
    assert "content-type" in response.headers

    if (
        expected_status == 200
        and "application/json" in response.headers["content-type"]
    ):
        # Should be valid JSON
        response.json()  # Will raise exception if invalid
