"""
Real GPU inference tests for PartPacker adapter.

Tests the PartPacker adapter with actual model inference on GPU.
No mocks - this performs real image-to-part-mesh generation.
"""

from pathlib import Path

import pytest
import torch

# Import the adapter
from adapters.partpacker_adapter import PartPackerImageToRawMeshAdapter
from tests.test_adapters.gpu_memory_tracker import track_gpu_memory


class TestPartPackerAdapter:
    """Test PartPacker adapter with real GPU inference"""

    @pytest.fixture
    def adapter(self):
        """Create a PartPacker adapter instance"""
        return PartPackerImageToRawMeshAdapter()

    @track_gpu_memory
    @pytest.mark.timeout(600)
    def test_image_to_part_mesh_generation(self, adapter):
        """Test image-to-part-mesh generation with real inference"""
        # Check if example image exists
        example_image = Path("assets/example_partpacker/barrel.png")
        if not example_image.exists():
            pytest.skip("Example PartPacker image not found")

        # Load the model
        adapter.load(0)

        try:
            inputs = {
                "image_path": str(example_image),
                "output_format": "glb",
                "remove_background": True,
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

            print(f"Successfully generated part mesh: {output_path}")
            print(
                f"Vertices: {gen_info['vertex_count']}, Faces: {gen_info['face_count']}"
            )

        finally:
            adapter.unload()

    @track_gpu_memory
    @pytest.mark.timeout(600)
    def test_different_output_formats(self, adapter):
        """Test generation with different output formats"""
        example_image = Path("assets/example_partpacker/cyan_car.png")
        if not example_image.exists():
            pytest.skip("Example PartPacker image not found")

        adapter.load(0)

        try:
            base_inputs = {
                "image_path": str(example_image),
                "remove_background": True,
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


class TestPartPackerErrorHandling:
    """Test error handling for PartPacker adapter"""

    @pytest.fixture
    def adapter(self):
        return PartPackerImageToRawMeshAdapter()

    def test_missing_image_file(self, adapter):
        """Test handling of missing image file"""
        adapter.load(0)

        try:
            inputs = {
                "image_path": "nonexistent_image.png",
                "output_format": "glb",
            }

            with pytest.raises(Exception):
                adapter.process(inputs)

            print("Missing image file correctly handled")

        finally:
            adapter.unload()

    def test_unsupported_format(self, adapter):
        """Test handling of unsupported output format"""
        example_image = Path("assets/example_partpacker/barrel.png")
        if not example_image.exists():
            pytest.skip("Example PartPacker image not found")

        adapter.load(0)

        try:
            inputs = {
                "image_path": str(example_image),
                "output_format": "stl",  # Not supported
            }

            with pytest.raises(Exception):
                adapter.process(inputs)

            print("Unsupported format correctly handled")

        finally:
            adapter.unload()


class TestPartPackerMemoryManagement:
    """Test GPU memory management for PartPacker adapter"""

    @track_gpu_memory
    def test_memory_cleanup(self):
        """Test memory cleanup after processing"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        adapter = PartPackerImageToRawMeshAdapter()

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
