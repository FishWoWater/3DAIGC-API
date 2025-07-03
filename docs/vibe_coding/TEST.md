# Testing Guide for 3D Generative Models Backend

This document describes the testing framework for the 3D Generative Models Backend, focusing on the `text-to-textured-mesh` feature.

## Test Structure

```
tests/
├── conftest.py                    # Test configuration and fixtures
├── run_tests.py                   # Test runner script
├── test_api/                      # API endpoint tests
│   ├── test_basic_endpoints.py    # Basic API functionality
│   └── test_text_to_textured_mesh.py  # Text-to-mesh API tests
├── test_adapters/                 # Adapter tests
│   └── test_trellis_adapter.py    # TRELLIS adapter tests
├── test_model/                    # Model abstraction tests
├── test_scheduler/                # Scheduler tests
└── requirements-test.txt          # Test dependencies
```

## Test Categories

### 1. Basic API Tests (`test_basic_endpoints.py`)

Tests fundamental API functionality:
- Health check endpoint
- Root endpoint with API information
- API documentation endpoints
- Error handling middleware
- CORS configuration
- Request/response processing

**Run with:**
```bash
python tests/run_tests.py basic
```

### 2. Text-to-Textured-Mesh API Tests (`test_text_to_textured_mesh.py`)

Tests the complete text-to-textured-mesh feature from a client perspective:
- Job submission and validation
- Job status checking
- Result retrieval and download
- Error handling
- End-to-end workflow

**Key test scenarios:**
- Valid request processing
- Input validation (prompt, quality, format, resolution)
- Job status progression (queued → processing → completed)
- Error conditions (scheduler failures, invalid inputs)
- File download functionality

**Run with:**
```bash
python tests/run_tests.py text-to-mesh
```

### 3. TRELLIS Adapter Tests (`test_trellis_adapter.py`)

Tests the TRELLIS adapter directly without API layer:
- Adapter initialization and configuration
- Model loading and unloading
- Text processing and mesh generation
- Environment setup and dependency management
- Error handling and edge cases

**Key test scenarios:**
- Adapter creation with various configurations
- Model loading success/failure scenarios
- Text processing with valid/invalid inputs
- Concurrent processing limits
- Memory cleanup and resource management
- Special characters and edge cases in prompts

**Run with:**
```bash
python tests/run_tests.py adapters
```

## Installation and Setup

### 1. Install Test Dependencies

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Or install specific packages
pip install pytest pytest-asyncio pytest-mock pytest-cov pytest-xdist httpx
```

### 2. Verify Installation

```bash
python tests/run_tests.py --check-deps
```

## Running Tests

### Quick Start

```bash
# Run all tests
python tests/run_tests.py

# Run specific test categories
python tests/run_tests.py api          # All API tests
python tests/run_tests.py adapters    # All adapter tests
python tests/run_tests.py basic       # Basic API tests only
python tests/run_tests.py text-to-mesh # Text-to-mesh specific tests
```

### Advanced Options

```bash
# Verbose output
python tests/run_tests.py -v

# Generate coverage report
python tests/run_tests.py -c

# Run tests in parallel
python tests/run_tests.py -p

# Combine options
python tests/run_tests.py text-to-mesh -v -c
```

### Direct pytest Usage

```bash
# Run all tests
pytest tests/

# Run specific test files
pytest tests/test_api/test_basic_endpoints.py -v
pytest tests/test_api/test_text_to_textured_mesh.py -v
pytest tests/test_adapters/test_trellis_adapter.py -v

# Run with markers
pytest -m "api and not slow" -v
pytest -m "unit" -v
pytest -m "integration" -v

# Generate coverage report
pytest --cov=api --cov=core --cov=adapters --cov-report=html tests/
```

## Test Configuration

### Markers

Tests are organized with markers for flexible execution:

- `unit`: Fast, isolated unit tests
- `integration`: Integration tests with real components
- `api`: API endpoint tests
- `adapter`: Adapter-specific tests
- `slow`: Tests that may take > 30s
- `gpu`: Tests requiring GPU access

### Fixtures

Common fixtures are provided in `conftest.py`:

- `temp_dir`: Temporary directory for test files
- `mock_scheduler`: Mock scheduler for API tests
- `mock_trellis_adapter`: Mock TRELLIS adapter
- `sample_job_request`: Sample job request object

### Mock Data

Test utilities provide:
- Sample prompts (valid/invalid)
- Test formats and quality settings
- Job status progressions
- Mock file system operations

## Test Environment

### Environment Variables

```bash
# Set test environment
export TESTING=true
export LOG_LEVEL=WARNING

# Disable real model loading in tests
export MOCK_ADAPTERS=true
```

### Configuration

Tests use separate configuration to avoid affecting the main application:
- Mock adapters instead of real models
- In-memory storage for temporary files
- Reduced logging verbosity
- Faster timeouts

## Debugging Tests

### Running Individual Tests

```bash
# Run single test method
pytest tests/test_api/test_text_to_textured_mesh.py::TestTextToTexturedMeshAPI::test_valid_text_to_textured_mesh_request -v

# Run single test class  
pytest tests/test_adapters/test_trellis_adapter.py::TestTrellisAdapterProcessing -v
```

### Debug Output

```bash
# Show print statements
pytest -s tests/test_api/test_basic_endpoints.py

# Show warnings
pytest --disable-warnings tests/

# Detailed traceback
pytest --tb=long tests/
```

### Test Coverage

```bash
# Generate HTML coverage report
pytest --cov=api --cov=core --cov=adapters --cov-report=html tests/

# View coverage in browser
open htmlcov/index.html
```

## Writing New Tests

### Test Naming Convention

- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`

### Example Test Structure

```python
"""
Module description and test scope.
"""

import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

class TestFeatureName:
    """Test specific feature or component"""
    
    def test_basic_functionality(self):
        """Test basic functionality works as expected"""
        # Arrange
        # Act  
        # Assert
        pass
    
    def test_error_handling(self):
        """Test error conditions are handled properly"""
        pass
    
    @pytest.mark.slow
    def test_performance_scenario(self):
        """Test performance-critical scenarios"""
        pass
```

### Mocking Guidelines

- Mock external dependencies (file system, network, GPU operations)
- Use `AsyncMock` for async methods
- Patch at the appropriate level (module vs class vs method)
- Verify mock calls when testing interactions

## Continuous Integration

### GitHub Actions Integration

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt -r requirements-test.txt
      - run: python tests/run_tests.py -c
```

### Quality Gates

- All tests must pass
- Coverage should be > 80%
- No critical security vulnerabilities
- Code style compliance

## Performance Testing

### Benchmarking

```bash
# Run performance tests
pytest -m "slow" --benchmark-only tests/

# Generate benchmark report
pytest --benchmark-json=benchmark.json tests/
```

### Load Testing

For API load testing, use external tools:
```bash
# Install locust for load testing
pip install locust

# Run load tests (if available)
locust -f tests/load_test.py --host http://localhost:8000
```

## Common Issues and Solutions

### Test Failures

1. **Import Errors**: Ensure PYTHONPATH includes the project root
2. **Async Test Issues**: Use `pytest-asyncio` and mark tests with `@pytest.mark.asyncio`
3. **Mock Problems**: Verify mock patches are at the correct import path
4. **File Path Issues**: Use `temp_dir` fixture for file operations

### Performance Issues

1. **Slow Tests**: Use `@pytest.mark.slow` and run separately
2. **Memory Leaks**: Ensure proper cleanup in teardown methods
3. **Parallel Execution**: Use `pytest-xdist` for parallel test execution

### Coverage Issues

1. **Low Coverage**: Add tests for uncovered code paths
2. **Exclude Non-Testable Code**: Use `pragma: no cover` for exceptional cases
3. **Integration vs Unit**: Balance between unit and integration test coverage

## Best Practices

1. **Test Isolation**: Each test should be independent
2. **Clear Assertions**: Use descriptive assertion messages
3. **Arrange-Act-Assert**: Structure tests clearly
4. **Mock Sparingly**: Mock only what's necessary
5. **Test Edge Cases**: Include boundary conditions and error scenarios
6. **Performance Awareness**: Mark slow tests appropriately
7. **Documentation**: Document complex test scenarios

## Support

For questions about testing:
1. Check this README first
2. Review existing test examples
3. Check the test output for specific error messages
4. Consult the pytest documentation for advanced usage
