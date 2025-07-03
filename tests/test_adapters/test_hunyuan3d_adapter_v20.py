"""
Real GPU inference tests for Hunyuan3D-2.0 adapters.

Tests the Hunyuan3D-2.0 adapters with actual model inference on GPU.
No mocks - this performs real image-to-mesh generation.
"""

from pathlib import Path

import pytest
import torch

# Import the adapters
from adapters.hunyuan3d_adapter_v20 import (
    Hunyuan3DV20ImageMeshPaintingAdapter,
    Hunyuan3DV20ImageToRawMeshAdapter,
    Hunyuan3DV20ImageToTexturedMeshAdapter,
)
from tests.test_adapters.gpu_memory_tracker import track_gpu_memory


class TestHunyuan3DV20RawMeshAdapter:
    """Test Hunyuan3D 2.0 raw mesh adapter with real GPU inference"""

    @pytest.fixture
    def adapter(self):
        """Create a Hunyuan3D 2.0 raw mesh adapter instance"""
        return Hunyuan3DV20ImageToRawMeshAdapter()

    @track_gpu_memory
    @pytest.mark.timeout(600)
    def test_image_to_raw_mesh_generation(self, adapter):
        """Test image-to-raw-mesh generation with real inference"""
        # Check if example image exists
        example_image = Path("assets/example_image/073.png")
        if not example_image.exists():
            pytest.skip("Example image not found")

        # Load the model
        adapter.load(0)

        try:
            inputs = {
                "image_path": str(example_image),
                "output_format": "glb",
                "num_inference_steps": 30,  # Reduced for faster testing
                "seed": 42,
            }

            result = adapter.process(inputs)

            # Verify the result
            assert result is not None
            assert "output_mesh_path" in result
            assert "generation_info" in result

            # Check the output file exists
            output_path = Path(result["output_mesh_path"])
            assert output_path.exists()
            assert output_path.suffix == ".glb"

            # Verify generation info
            gen_info = result["generation_info"]
            assert "vertex_count" in gen_info
            assert "face_count" in gen_info
            assert gen_info["has_texture"] == False
            assert gen_info["num_inference_steps"] == 30

            print(f"Successfully generated raw mesh: {output_path}")
            print(
                f"Vertices: {gen_info['vertex_count']}, Faces: {gen_info['face_count']}"
            )

        finally:
            adapter.unload()

    @track_gpu_memory
    @pytest.mark.timeout(600)
    def test_different_output_formats(self, adapter):
        """Test generation with different output formats"""
        example_image = Path("assets/example_image/075.png")
        if not example_image.exists():
            pytest.skip("Example image not found")

        adapter.load(0)

        try:
            base_inputs = {
                "image_path": str(example_image),
                "num_inference_steps": 20,  # Reduced for faster testing
                "seed": 42,
            }

            for output_format in ["glb", "obj"]:
                inputs = {**base_inputs, "output_format": output_format}

                result = adapter.process(inputs)

                assert result is not None
                output_path = Path(result["output_mesh_path"])
                assert output_path.exists()
                assert output_path.suffix == f".{output_format}"

                print(f"Generated {output_format} format: {output_path}")

        finally:
            adapter.unload()

    @track_gpu_memory
    @pytest.mark.timeout(600)
    def test_custom_parameters(self, adapter):
        """Test generation with custom parameters"""
        example_image = Path("assets/example_image/1687.png")
        if not example_image.exists():
            pytest.skip("Example image not found")

        adapter.load(0)

        try:
            inputs = {
                "image_path": str(example_image),
                "output_format": "glb",
                "num_inference_steps": 25,
                "octree_resolution": 320,  # Lower resolution for faster testing
                "num_chunks": 15000,  # Reduced for faster testing
                "seed": 123,
            }

            result = adapter.process(inputs)

            assert result is not None
            assert "generation_info" in result

            gen_info = result["generation_info"]
            assert gen_info["num_inference_steps"] == 25
            assert gen_info["octree_resolution"] == 320
            assert gen_info["seed"] == 123

            print(f"Generated with custom parameters: {result['output_mesh_path']}")

        finally:
            adapter.unload()


class TestHunyuan3DV20TexturedMeshAdapter:
    """Test Hunyuan3D 2.0 textured mesh adapter with real GPU inference"""

    @pytest.fixture
    def adapter(self):
        """Create a Hunyuan3D 2.0 textured mesh adapter instance"""
        return Hunyuan3DV20ImageToTexturedMeshAdapter()

    @track_gpu_memory
    @pytest.mark.timeout(1600)
    def test_image_to_textured_mesh_generation(self, adapter):
        """Test image-to-textured-mesh generation with real inference"""
        example_image = Path("assets/example_image/1687.png")
        if not example_image.exists():
            pytest.skip("Example image not found")

        adapter.load(0)

        try:
            inputs = {
                "image_path": str(example_image),
                "output_format": "glb",
                "num_inference_steps": 25,  # Reduced for faster testing
                "seed": 42,
            }

            result = adapter.process(inputs)

            # Verify the result
            assert result is not None
            assert "output_mesh_path" in result
            assert "generation_info" in result

            # Check the output file exists
            output_path = Path(result["output_mesh_path"])
            assert output_path.exists()
            assert output_path.suffix == ".glb"

            # Verify generation info
            gen_info = result["generation_info"]
            assert gen_info["has_texture"] == True
            assert gen_info["num_inference_steps"] == 25

            print(f"Successfully generated textured mesh: {output_path}")
            print(
                f"Vertices: {gen_info['vertex_count']}, Faces: {gen_info['face_count']}"
            )

        finally:
            adapter.unload()


class TestHunyuan3DV20MeshPaintingAdapter:
    """Test Hunyuan3D 2.0 mesh painting adapter with real GPU inference"""

    @pytest.fixture
    def adapter(self):
        """Create a Hunyuan3D 2.0 mesh painting adapter instance"""
        return Hunyuan3DV20ImageMeshPaintingAdapter()

    @track_gpu_memory
    @pytest.mark.timeout(600)
    def test_mesh_painting(self, adapter):
        """Test mesh painting with real inference"""
        # Check if example assets exist
        example_mesh = Path("assets/example_mesh/typical_creature_dragon.obj")
        example_image = Path("assets/example_image/073.png")

        if not example_mesh.exists() or not example_image.exists():
            pytest.skip("Example mesh or image not found")

        adapter.load(0)

        try:
            inputs = {
                "mesh_path": str(example_mesh),
                "image_path": str(example_image),
                "output_format": "glb",
                "num_inference_steps": 20,  # Reduced for faster testing
            }

            result = adapter.process(inputs)

            # Verify the result
            assert result is not None
            assert "output_mesh_path" in result
            assert "painting_info" in result

            # Check the output file exists
            output_path = Path(result["output_mesh_path"])
            assert output_path.exists()
            assert output_path.suffix == ".glb"

            # Verify painting info
            paint_info = result["painting_info"]
            assert paint_info["has_texture"] == True
            assert paint_info["num_inference_steps"] == 20

            print(f"Successfully painted mesh: {output_path}")

        finally:
            adapter.unload()


class TestHunyuan3DV20ErrorHandling:
    """Test error handling for Hunyuan3D 2.0 adapters"""

    @pytest.fixture
    def raw_adapter(self):
        """Create raw mesh adapter instance"""
        return Hunyuan3DV20ImageToRawMeshAdapter()

    @pytest.fixture
    def paint_adapter(self):
        """Create mesh painting adapter instance"""
        return Hunyuan3DV20ImageMeshPaintingAdapter()

    def test_missing_image_file(self, raw_adapter):
        """Test handling of missing image file"""
        raw_adapter.load(0)

        try:
            inputs = {
                "image_path": "nonexistent_image.png",
                "output_format": "glb",
            }

            with pytest.raises(Exception):
                raw_adapter.process(inputs)

            print("Missing image file correctly handled")

        finally:
            raw_adapter.unload()

    def test_missing_mesh_file(self, paint_adapter):
        """Test handling of missing mesh file for painting adapter"""
        paint_adapter.load(0)

        try:
            inputs = {
                "mesh_path": "nonexistent_mesh.obj",
                "image_path": "test_image.png",
                "output_format": "glb",
            }

            with pytest.raises(Exception):
                paint_adapter.process(inputs)

            print("Missing mesh file correctly handled")

        finally:
            paint_adapter.unload()

    def test_unsupported_format(self, raw_adapter):
        """Test handling of unsupported output format"""
        example_image = Path("assets/example_image/073.png")
        if not example_image.exists():
            pytest.skip("Example image not found")

        raw_adapter.load(0)

        try:
            inputs = {
                "image_path": str(example_image),
                "output_format": "stl",  # Not supported
            }

            with pytest.raises(Exception):
                raw_adapter.process(inputs)

            print("Unsupported format correctly handled")

        finally:
            raw_adapter.unload()

    def test_processing_without_loading(self, raw_adapter):
        """Test error when processing without loading model"""
        inputs = {
            "image_path": "test.png",
            "output_format": "glb",
        }

        with pytest.raises(Exception):
            raw_adapter.process(inputs)

        print("Processing without loading correctly handled")


class TestHunyuan3DV20MemoryManagement:
    """Test GPU memory management for Hunyuan3D 2.0 adapters"""

    @track_gpu_memory
    def test_memory_cleanup_raw_mesh(self):
        """Test memory cleanup after raw mesh generation"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        adapter = Hunyuan3DV20ImageToRawMeshAdapter()

        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()

        adapter.load(0)

        try:
            # Memory should increase after loading
            loaded_memory = torch.cuda.memory_allocated()
            assert loaded_memory > initial_memory
            print(f"Memory after loading: {loaded_memory / 1024**2:.1f}MB")

        finally:
            adapter.unload()

        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()

        print(
            f"Memory usage - Initial: {initial_memory / 1024**2:.1f}MB, Final: {final_memory / 1024**2:.1f}MB"
        )

        # Memory should be released (allow some tolerance)
        assert final_memory <= initial_memory + 100 * 1024**2  # 100MB tolerance


class TestHunyuan3DV20ModelCompatibility:
    """Test model compatibility and format support"""

    def test_supported_formats(self):
        """Test supported input and output formats"""
        adapter = Hunyuan3DV20ImageToRawMeshAdapter()

        # Test supported formats
        formats = adapter.get_supported_formats()
        assert "input" in formats
        assert "output" in formats
        assert "png" in formats["input"]
        assert "jpg" in formats["input"]
        assert "glb" in formats["output"]
        assert "obj" in formats["output"]

        print("Supported formats validated")

    @track_gpu_memory
    def test_vram_requirements(self):
        """Test VRAM requirements and availability"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        adapter = Hunyuan3DV20ImageToRawMeshAdapter()

        # Check available VRAM
        available_vram = torch.cuda.get_device_properties(0).total_memory
        print(f"Available VRAM: {available_vram / 1024**2:.0f}MB")

        # Test basic GPU functionality by loading adapter
        adapter.load(0)
        try:
            print("Adapter loaded successfully")
        finally:
            adapter.unload()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
