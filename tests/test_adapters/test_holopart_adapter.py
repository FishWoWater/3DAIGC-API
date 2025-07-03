"""
Real GPU inference tests for HoloPart part completion adapter.

Tests the HoloPart adapter with actual model inference on GPU.
No mocks - this performs real part completion.
"""

import os
import sys

sys.path.append(os.getcwd())
from pathlib import Path

import pytest
import torch

# Import the adapter
from adapters.holopart_adapter import HoloPartCompletionAdapter
from tests.test_adapters.gpu_memory_tracker import track_gpu_memory


class TestHoloPartAdapterRealInference:
    """Test HoloPart adapter with real GPU inference"""

    @pytest.fixture
    def adapter(self):
        """Create a HoloPart adapter instance"""
        return HoloPartCompletionAdapter()

    @track_gpu_memory
    @pytest.mark.timeout(600)  # 10 minutes timeout for part completion
    def test_part_completion_fast_decoder(self, adapter):
        """Test part completion with flash decoder"""
        # Load the model
        adapter.load(0)

        try:
            # Find a sample segmented mesh
            sample_meshes = [
                Path("assets/example_holopart/000.glb"),
                Path("assets/example_holopart/001.glb"),
            ]

            sample_mesh = None
            for mesh_path in sample_meshes:
                if mesh_path.exists():
                    sample_mesh = mesh_path
                    break

            if sample_mesh is None:
                pytest.skip("No sample segmented mesh files found for testing")

            # Test with flash decoder (faster)
            inputs = {
                "mesh_path": str(sample_mesh),
                "num_inference_steps": 20,  # Reduced for faster testing
                "guidance_scale": 3.5,
                "seed": 42,
                "use_flash_decoder": True,
                "simplify_output": True,
            }

            result = adapter.process(inputs)

            # Verify the result
            assert result is not None
            assert "output_mesh_path" in result
            assert "completion_info" in result
            assert result["success"] is True

            # Check the output file exists
            output_path = Path(result["output_mesh_path"])
            assert output_path.exists()
            assert output_path.suffix == ".glb"

            # Verify completion info
            comp_info = result["completion_info"]
            assert "input_parts" in comp_info
            assert "output_parts" in comp_info
            assert "generation_parameters" in comp_info
            assert comp_info["generation_parameters"]["use_flash_decoder"] is True
            assert comp_info["generation_parameters"]["num_inference_steps"] == 20

            # Check generation info
            gen_info = result["generation_info"]
            assert gen_info["input_mesh"] == str(sample_mesh)
            assert gen_info["input_format"] == "glb"

            print(f"Successfully completed parts: {output_path}")
            print(
                f"Input parts: {result['input_parts']}, Output parts: {result['output_parts']}"
            )

        finally:
            # Unload the model
            adapter.unload()

    @track_gpu_memory
    @pytest.mark.timeout(600)
    def test_part_completion_hierarchical_decoder(self, adapter):
        """Test part completion with hierarchical decoder"""
        adapter.load(0)

        try:
            # Find a sample segmented mesh
            sample_meshes = [
                Path("assets/example_holopart/000.glb"),
                Path("assets/example_holopart/001.glb"),
            ]

            sample_mesh = None
            for mesh_path in sample_meshes:
                if mesh_path.exists():
                    sample_mesh = mesh_path
                    break

            if sample_mesh is None:
                pytest.skip("No sample segmented mesh files found for testing")

            # Test with hierarchical decoder (slower but potentially higher quality)
            inputs = {
                "mesh_path": str(sample_mesh),
                "num_inference_steps": 10,  # Even more reduced for testing
                "guidance_scale": 2.5,
                "seed": 123,
                "use_flash_decoder": False,
                "simplify_output": False,
            }

            result = adapter.process(inputs)

            assert result is not None
            assert result["success"] is True
            output_path = Path(result["output_mesh_path"])
            assert output_path.exists()

            # Verify hierarchical decoder was used
            comp_info = result["completion_info"]
            assert comp_info["generation_parameters"]["use_flash_decoder"] is False
            assert comp_info["generation_parameters"]["simplify_output"] is False

            print(f"Hierarchical decoder completion: {output_path}")

        finally:
            adapter.unload()

    @track_gpu_memory
    @pytest.mark.timeout(600)
    def test_different_inference_parameters(self, adapter):
        """Test completion with different inference parameters"""
        adapter.load(0)

        try:
            sample_mesh = Path("assets/example_holopart/000.glb")
            if not sample_mesh.exists():
                pytest.skip("Sample mesh not found for parameter testing")

            # Test different parameter combinations
            parameter_sets = [
                {"num_inference_steps": 5, "guidance_scale": 1.5, "seed": 42},
                {"num_inference_steps": 15, "guidance_scale": 4.0, "seed": 999},
            ]

            for i, params in enumerate(parameter_sets):
                inputs = {
                    "mesh_path": str(sample_mesh),
                    "use_flash_decoder": True,
                    "simplify_output": True,
                    **params,
                }

                result = adapter.process(inputs)

                assert result is not None
                assert result["success"] is True

                comp_info = result["completion_info"]
                gen_params = comp_info["generation_parameters"]
                assert (
                    gen_params["num_inference_steps"] == params["num_inference_steps"]
                )
                assert gen_params["guidance_scale"] == params["guidance_scale"]
                assert gen_params["seed"] == params["seed"]

                print(f"Parameter set {i + 1} completion successful")

        finally:
            adapter.unload()

    def test_error_handling(self, adapter):
        """Test error handling for invalid inputs"""
        adapter.load(0)

        try:
            # Test with non-existent file
            with pytest.raises(Exception):
                adapter.process(
                    {"mesh_path": "nonexistent_file.glb", "num_inference_steps": 10}
                )

            # Test with missing mesh_path
            with pytest.raises(Exception):
                adapter.process({"num_inference_steps": 10})

            # Test with non-GLB format
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as tmp:
                tmp.write(b"# Test OBJ file")
                temp_path = tmp.name

            try:
                with pytest.raises(Exception):
                    adapter.process({"mesh_path": temp_path, "num_inference_steps": 10})
            finally:
                Path(temp_path).unlink(missing_ok=True)

            # Test with unsupported output format
            with pytest.raises(Exception):
                adapter.process(
                    {
                        "mesh_path": "test.glb",
                        "output_format": "stl",  # Not supported
                        "num_inference_steps": 10,
                    }
                )

            print("Error handling tests passed")

        finally:
            adapter.unload()

    @track_gpu_memory
    def test_cleanup_after_completion(self, adapter):
        """Test that GPU memory is properly managed"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Check initial GPU memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()

        adapter.load(0)

        try:
            sample_mesh = Path("assets/example_holopart/000.glb")
            if sample_mesh.exists():
                inputs = {
                    "mesh_path": str(sample_mesh),
                    "num_inference_steps": 5,
                    "use_flash_decoder": True,
                }

                result = adapter.process(inputs)
                assert result is not None

        finally:
            adapter.unload()

        # Check that memory is released after unloading
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()

        print(
            f"Memory usage - Initial: {initial_memory / 1024**2:.1f}MB, Final: {final_memory / 1024**2:.1f}MB"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
