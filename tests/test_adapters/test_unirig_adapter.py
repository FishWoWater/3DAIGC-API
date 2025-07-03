"""
Comprehensive tests for UniRig adapter.

This module tests the UniRig auto-rigging adapter with real GPU inference,
covering skeleton generation, skin weight computation, and full pipeline modes.
"""

import logging
import time
from pathlib import Path

import pytest
import torch

from adapters.unirig_adapter import UniRigAdapter
from tests.test_adapters.gpu_memory_tracker import track_gpu_memory

# Set up logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
TEST_ASSET_DIR_OBJ = Path("assets/example_mesh")
TEST_ASSET_DIR_SKELETON = Path("assets/example_autorig/skeleton")
TEST_OUTPUT_DIR = Path("outputs/test_unirig")
SAMPLE_MESH_FILES = [
    "typical_creature_dragon.obj",
    "typical_creature_elephant.obj",
    "typical_creature_furry.obj",
]
SAMPLE_SKELETON_FILES = [
    "giraffe.fbx",
]

# Skip tests if CUDA not available
CUDA_AVAILABLE = torch.cuda.is_available()
SKIP_REASON = "CUDA not available - UniRig requires GPU"


class TestUniRigAdapter:
    """Comprehensive tests for UniRig adapter functionality."""

    @pytest.fixture(scope="class", autouse=True)
    def setup_class(self):
        """Set up test environment."""
        # Create test output directory
        TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Verify test assets exist
        if not TEST_ASSET_DIR_OBJ.exists():
            pytest.skip(f"Test asset directory not found: {TEST_ASSET_DIR_OBJ}")

        # Check for at least one test mesh
        available_meshes = []
        for mesh_file in SAMPLE_MESH_FILES:
            mesh_path = TEST_ASSET_DIR_OBJ / mesh_file
            if mesh_path.exists():
                available_meshes.append(mesh_path)

        if not available_meshes:
            pytest.skip(f"No test mesh files found in {TEST_ASSET_DIR_OBJ}")

        logger.info(f"Found {len(available_meshes)} test meshes")
        yield

        # Cleanup after tests (optional - uncomment if you want cleanup)
        # if TEST_OUTPUT_DIR.exists():
        #     shutil.rmtree(TEST_OUTPUT_DIR)

    @pytest.fixture
    def adapter(self):
        """Create UniRig adapter instance."""
        return UniRigAdapter(
            model_id="test_unirig",
            vram_requirement=6144,  # Reduced for testing
            device="cuda" if CUDA_AVAILABLE else "cpu",
        )

    @pytest.fixture
    def sample_mesh(self):
        """Get path to a sample mesh file."""
        for mesh_file in SAMPLE_MESH_FILES:
            mesh_path = TEST_ASSET_DIR_OBJ / mesh_file
            if mesh_path.exists():
                return mesh_path
        pytest.skip("No sample mesh files available")

    @pytest.fixture
    def sample_skeleton(self):
        """Get path to a sample skeleton file."""
        for skeleton_file in SAMPLE_SKELETON_FILES:
            skeleton_path = TEST_ASSET_DIR_SKELETON / skeleton_file
            if skeleton_path.exists():
                return skeleton_path
        pytest.skip("No sample skeleton files available")

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_adapter_initialization(self, adapter):
        """Test adapter initialization and configuration."""
        assert adapter.model_id == "test_unirig"
        assert adapter.vram_requirement == 6144
        assert adapter.device in ["cuda", "cpu"]

        # Test supported formats
        formats = adapter.get_supported_formats()
        assert "input" in formats
        assert "output" in formats
        assert "fbx" in formats["input"]
        assert "fbx" in formats["output"]

        # Test model info
        info = adapter.get_model_info()
        assert "capabilities" in info
        assert "supported_modes" in info
        assert len(info["supported_modes"]) == 3

    @track_gpu_memory
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_model_loading(self, adapter):
        """Test model loading and unloading."""
        # Test loading
        start_time = time.time()
        model = adapter.load(gpu_id=0)
        load_time = time.time() - start_time

        assert model is not None
        assert adapter.inference_engine is not None
        logger.info(f"Model loading took {load_time:.2f} seconds")

        # Test unloading
        success = adapter.unload()
        assert success
        assert adapter.inference_engine is None

    @track_gpu_memory
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_skeleton_generation(self, adapter, sample_mesh):
        """Test skeleton generation mode."""
        # Load model
        adapter.load(gpu_id=0)

        try:
            # Test skeleton generation
            inputs = {
                "mesh_path": str(sample_mesh),
                "rig_mode": "skeleton",
                "output_format": "fbx",
            }

            start_time = time.time()
            result = adapter.process(inputs)
            process_time = time.time() - start_time

            # Validate result
            assert result["success"] is True
            assert "output_mesh_path" in result
            assert result["rig_info"]["rig_mode"] == "skeleton"
            assert result["rig_info"]["has_skinning"] is False
            assert result["rig_info"]["skeleton_only"] is True
            assert result["bone_count"] > 0

            # Verify output file exists
            output_path = Path(result["output_mesh_path"])
            assert output_path.exists()
            assert output_path.suffix.lower() == ".fbx"

            logger.info(f"Skeleton generation took {process_time:.2f} seconds")
            logger.info(f"Generated {result['bone_count']} bones")

        finally:
            adapter.unload()

    @track_gpu_memory
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_skin_generation(self, adapter, sample_skeleton):
        """Test skin weight generation mode."""
        # Load model
        adapter.load(gpu_id=0)

        try:
            # Test skin weight generation
            inputs = {
                "mesh_path": str(sample_skeleton),
                "rig_mode": "skin",
                "output_format": "fbx",
            }

            start_time = time.time()
            result = adapter.process(inputs)
            process_time = time.time() - start_time

            # Validate result
            assert result["success"] is True
            assert "output_mesh_path" in result
            assert result["rig_info"]["rig_mode"] == "skin"
            assert result["rig_info"]["has_skinning"] is True
            assert result["rig_info"]["skeleton_only"] is False
            assert result["bone_count"] > 0

            # Verify output file exists
            output_path = Path(result["output_mesh_path"])
            assert output_path.exists()
            assert output_path.suffix.lower() == ".fbx"

            logger.info(f"Skin generation took {process_time:.2f} seconds")
            logger.info(f"Generated skinning for {result['bone_count']} bones")

        finally:
            adapter.unload()

    @track_gpu_memory
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_full_pipeline(self, adapter, sample_mesh):
        """Test full rigging pipeline (skeleton + skin)."""
        # Load model
        adapter.load(gpu_id=0)

        try:
            # Test full pipeline
            inputs = {
                "mesh_path": str(sample_mesh),
                "rig_mode": "full",
                "output_format": "fbx",
                "with_skinning": True,
            }

            start_time = time.time()
            result = adapter.process(inputs)
            process_time = time.time() - start_time

            # Validate result
            assert result["success"] is True
            assert "output_mesh_path" in result
            assert result["rig_info"]["rig_mode"] == "full"
            assert result["rig_info"]["has_skinning"] is True
            assert result["bone_count"] > 0

            # Verify output file exists and is larger (contains both skeleton and skin)
            output_path = Path(result["output_mesh_path"])
            assert output_path.exists()
            assert output_path.suffix.lower() == ".fbx"
            assert output_path.stat().st_size > 1024  # At least 1KB

            logger.info(f"Full pipeline took {process_time:.2f} seconds")
            logger.info(f"Generated complete rig with {result['bone_count']} bones")

        finally:
            adapter.unload()

    @track_gpu_memory
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_different_output_formats(self, adapter, sample_mesh):
        """Test different output formats."""
        adapter.load(gpu_id=0)

        try:
            formats_to_test = ["fbx", "obj", "glb"]

            for output_format in formats_to_test:
                inputs = {
                    "mesh_path": str(sample_mesh),
                    "rig_mode": "skeleton",  # Use skeleton mode for faster testing
                    "output_format": output_format,
                }

                result = adapter.process(inputs)

                # Validate result
                assert result["success"] is True
                assert result["format"] == output_format

                # Verify output file has correct extension
                output_path = Path(result["output_mesh_path"])
                assert output_path.exists()
                assert output_path.suffix.lower() == f".{output_format.lower()}"

                logger.info(f"Successfully generated {output_format} format")

        finally:
            adapter.unload()

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_error_handling(self, adapter):
        """Test error handling for invalid inputs."""
        adapter.load(gpu_id=0)

        try:
            # Test missing mesh_path
            with pytest.raises(Exception) as exc_info:
                adapter.process({})
            assert "mesh_path is required" in str(exc_info.value)

            # Test non-existent file
            with pytest.raises(Exception) as exc_info:
                adapter.process({"mesh_path": "non_existent_file.obj"})
            assert "not found" in str(exc_info.value).lower()

            # Test invalid rig mode
            with pytest.raises(Exception) as exc_info:
                adapter.process(
                    {
                        "mesh_path": str(TEST_ASSET_DIR_OBJ / SAMPLE_MESH_FILES[0]),
                        "rig_mode": "invalid_mode",
                    }
                )
            assert "Invalid rig_mode" in str(exc_info.value)

            logger.info("Error handling tests passed")

        finally:
            adapter.unload()

    @track_gpu_memory
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_memory_management(self, adapter, sample_mesh):
        """Test memory management during processing."""
        # Get initial GPU memory
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated()

        # Load and process
        adapter.load(gpu_id=0)

        # Process multiple times to check for memory leaks
        for i in range(3):
            inputs = {
                "mesh_path": str(sample_mesh),
                "rig_mode": "skeleton",
                "output_format": "fbx",
            }

            result = adapter.process(inputs)
            assert result["success"] is True

            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated()
                logger.info(
                    f"Iteration {i + 1}: GPU memory used: {current_memory / 1024**2:.1f} MB"
                )

        # Unload and check memory cleanup
        adapter.unload()

        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            logger.info(f"Final GPU memory: {final_memory / 1024**2:.1f} MB")

            # Memory should be cleaned up (allowing some tolerance)
            memory_increase = final_memory - initial_memory
            assert memory_increase < 500 * 1024**2  # Less than 500MB increase

    def test_adapter_without_gpu(self):
        """Test adapter behavior when GPU is not available."""
        # This test runs regardless of GPU availability
        adapter = UniRigAdapter(device="cpu")

        # Should be able to initialize
        assert adapter.device == "cpu"

        # Loading might fail on CPU-only setup (which is expected)
        # This tests graceful error handling
        try:
            adapter.load(gpu_id=0)
            # If it loads, try a simple operation
            adapter.unload()
        except Exception as e:
            # Expected for CPU-only environments
            logger.info(f"CPU-only test failed as expected: {str(e)}")

    def test_configuration_validation(self):
        """Test adapter configuration validation."""
        # Test valid configuration
        adapter = UniRigAdapter(model_id="test_unirig", vram_requirement=4096)
        assert adapter.model_id == "test_unirig"
        assert adapter.vram_requirement == 4096

        # Test invalid UniRig root path
        with pytest.raises(FileNotFoundError):
            UniRigAdapter(unirig_root="/invalid/path")


# Integration test function for running all tests
def run_all_tests():
    """Run all UniRig adapter tests."""
    if not CUDA_AVAILABLE:
        logger.warning("CUDA not available - skipping GPU tests")
        return

    logger.info("Starting UniRig adapter integration tests...")

    # Create adapter
    adapter = UniRigAdapter()

    # Find test mesh
    sample_mesh = None
    for mesh_file in SAMPLE_MESH_FILES:
        mesh_path = TEST_ASSET_DIR_OBJ / mesh_file
        if mesh_path.exists():
            sample_mesh = mesh_path
            break

    if not sample_mesh:
        logger.error("No test mesh files found")
        return

    try:
        # Test model loading
        logger.info("Testing model loading...")
        adapter.load(gpu_id=0)

        # Test skeleton generation
        logger.info("Testing skeleton generation...")
        result = adapter.process(
            {
                "mesh_path": str(sample_mesh),
                "rig_mode": "skeleton",
                "output_format": "fbx",
            }
        )
        logger.info(f"Skeleton generation result: {result['success']}")

        # Test full pipeline
        logger.info("Testing full pipeline...")
        result = adapter.process(
            {"mesh_path": str(sample_mesh), "rig_mode": "full", "output_format": "fbx"}
        )
        logger.info(f"Full pipeline result: {result['success']}")

        logger.info("All tests completed successfully!")

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise
    finally:
        adapter.unload()


if __name__ == "__main__":
    # Run integration tests directly
    run_all_tests()
