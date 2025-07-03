"""
Real GPU inference tests for Hunyuan3D-2.1 adapters.

Tests the Hunyuan3D-2.1 adapters with actual model inference on GPU.
No mocks - this performs real image-to-mesh generation.
"""

from pathlib import Path

import pytest
import torch

# Import the adapters
from adapters.hunyuan3d_adapter_v21 import (
    Hunyuan3DV21ImageMeshPaintingAdapter,
    Hunyuan3DV21ImageToRawMeshAdapter,
    Hunyuan3DV21ImageToTexturedMeshAdapter,
)
from tests.test_adapters.gpu_memory_tracker import track_gpu_memory


class TestHunyuan3DRawMeshAdapter:
    """Test Hunyuan3D raw mesh adapter with real GPU inference"""

    @pytest.fixture
    def adapter(self):
        """Create a Hunyuan3D raw mesh adapter instance"""
        return Hunyuan3DV21ImageToRawMeshAdapter()

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
            # assert gen_info["model"] == "hunyuan3d-2.1-raw"
            assert "vertex_count" in gen_info
            assert "face_count" in gen_info
            assert not gen_info["has_texture"]

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
            base_inputs = {"image_path": str(example_image), "seed": 42}

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


class TestHunyuan3DTexturedMeshAdapter:
    """Test Hunyuan3D textured mesh adapter with real GPU inference"""

    @pytest.fixture
    def adapter(self):
        """Create a Hunyuan3D textured mesh adapter instance"""
        return Hunyuan3DV21ImageToTexturedMeshAdapter()

    @track_gpu_memory
    @pytest.mark.timeout(1200)
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
                "max_num_view": 4,  # Reduced for faster testing
                "resolution": 256,  # Lower resolution for faster testing
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
            # assert gen_info["model"] == "hunyuan3d-2.1-textured"
            assert gen_info["has_texture"]
            assert gen_info["max_num_view"] == 4
            assert gen_info["resolution"] == 256

            print(f"Successfully generated textured mesh: {output_path}")
            print(
                f"Vertices: {gen_info['vertex_count']}, Faces: {gen_info['face_count']}"
            )

        finally:
            adapter.unload()


class TestHunyuan3DMeshPaintingAdapter:
    """Test Hunyuan3D mesh painting adapter with real GPU inference"""

    @pytest.fixture
    def adapter(self):
        """Create a Hunyuan3D mesh painting adapter instance"""
        return Hunyuan3DV21ImageMeshPaintingAdapter()

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
                "max_num_view": 4,  # Reduced for faster testing
                "resolution": 256,  # Lower resolution for faster testing
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
            assert paint_info["model"] == "hunyuan3d-2.1-painting"
            assert paint_info["max_num_view"] == 4
            assert paint_info["resolution"] == 256

            print(f"Successfully painted mesh: {output_path}")

        finally:
            adapter.unload()


class TestHunyuan3DErrorHandling:
    """Test error handling for Hunyuan3D adapters"""

    @pytest.fixture
    def raw_adapter(self):
        """Create raw mesh adapter for error testing"""
        return Hunyuan3DV21ImageToRawMeshAdapter()

    @pytest.fixture
    def paint_adapter(self):
        """Create painting adapter for error testing"""
        return Hunyuan3DV21ImageMeshPaintingAdapter()

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
        """Test handling of missing mesh file for painting"""
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


class TestHunyuan3DMemoryManagement:
    """Test GPU memory management for Hunyuan3D adapters"""

    @track_gpu_memory
    def test_memory_cleanup_raw_mesh(self):
        """Test memory cleanup after raw mesh generation"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        adapter = Hunyuan3DV21ImageToRawMeshAdapter()

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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
