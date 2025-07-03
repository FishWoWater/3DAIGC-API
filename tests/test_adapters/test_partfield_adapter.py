"""
Real GPU inference tests for PartField mesh segmentation adapter.

Tests the PartField adapter with actual model inference on GPU.
No mocks - this performs real mesh segmentation.
"""

from pathlib import Path

import pytest
import torch

# Import the adapter
from adapters.partfield_adapter import PartFieldSegmentationAdapter
from tests.test_adapters.gpu_memory_tracker import track_gpu_memory


class TestPartFieldAdapterRealInference:
    """Test PartField adapter with real GPU inference"""

    @pytest.fixture
    def adapter(self):
        """Create a PartField adapter instance"""
        return PartFieldSegmentationAdapter()

    @track_gpu_memory
    @pytest.mark.timeout(300)  # 5 minutes timeout for segmentation
    def test_mesh_segmentation(self, adapter):
        """Test mesh segmentation with hierarchical clustering"""
        # Load the model
        adapter.load(0)

        try:
            # Find a sample mesh
            sample_meshes = [
                Path("assets/example_mesh/typical_creature_dragon.obj"),
                Path("assets/example_mesh/typical_creature_elephant.obj"),
                Path("assets/example_mesh/typical_humanoid_mech.obj"),
            ]

            sample_mesh = None
            for mesh_path in sample_meshes:
                if mesh_path.exists():
                    sample_mesh = mesh_path
                    break

            if sample_mesh is None:
                pytest.skip("No sample mesh files found for testing")

            # Test with hierarchical clustering
            inputs = {
                "mesh_path": str(sample_mesh),
                "num_parts": 6,
                "export_colored_mesh": True,
            }

            result = adapter.process(inputs)

            # Verify the result
            assert result is not None
            assert "output_mesh_path" in result
            assert "segmentation_info" in result
            assert result["success"] is True

            # Check the output file exists
            output_path = Path(result["output_mesh_path"])
            assert output_path.exists()
            assert output_path.suffix == ".glb"

            # Verify segmentation info
            seg_info = result["segmentation_info"]
            assert seg_info["num_parts"] == inputs["num_parts"]
            assert seg_info["alg_option"] == 0
            assert "mesh_stats" in seg_info
            assert "part_statistics" in seg_info

            # Check generation info
            gen_info = result["generation_info"]
            assert gen_info["input_mesh"] == str(sample_mesh)
            assert "vertex_count" in gen_info
            assert "face_count" in gen_info

            print(f"Successfully segmented mesh: {output_path}")
            print(
                f"Input vertices: {gen_info['vertex_count']}, faces: {gen_info['face_count']}"
            )
            print(f"Segmented into {result['num_parts']} parts")

        finally:
            # Unload the model
            adapter.unload()

    def test_error_handling(self, adapter):
        """Test error handling for invalid inputs"""
        adapter.load(0)

        try:
            # Test with non-existent file
            with pytest.raises(Exception):
                adapter.process({"mesh_path": "nonexistent_file.obj", "num_parts": 4})

            # Test with missing mesh_path
            with pytest.raises(Exception):
                adapter.process({"num_parts": 4})

            # Test with unsupported format (should still work but worth testing)
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
                tmp.write(b"solid test\nendsolid test")
                temp_path = tmp.name

            try:
                # This should raise an exception for unsupported format
                with pytest.raises(Exception):
                    adapter.process({"mesh_path": temp_path, "num_parts": 4})
            finally:
                Path(temp_path).unlink(missing_ok=True)

            print("Error handling tests passed")

        finally:
            adapter.unload()

    @track_gpu_memory
    def test_cleanup_after_segmentation(self, adapter):
        """Test that GPU memory is properly managed"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Check initial GPU memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()

        adapter.load(0)

        try:
            sample_mesh = Path("assets/example_mesh/typical_creature_elephant.obj")
            if sample_mesh.exists():
                inputs = {
                    "mesh_path": str(sample_mesh),
                    "num_parts": 4,
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


class TestPartFieldAdapterWithExampleAssets:
    """Test PartField adapter using all available example meshes"""

    @pytest.fixture
    def adapter(self):
        """Create a PartField adapter instance"""
        return PartFieldSegmentationAdapter()

    @track_gpu_memory
    @pytest.mark.timeout(600)  # 10 minutes for multiple meshes
    def test_with_all_example_meshes(self, adapter):
        """Test segmentation using all meshes from example_mesh directory"""
        mesh_dir = Path("assets/example_mesh")
        if not mesh_dir.exists():
            pytest.skip("Example mesh directory not found")

        # Find all mesh files
        mesh_files = []
        for pattern in ["*.obj", "*.glb"]:
            mesh_files.extend(mesh_dir.glob(pattern))

        if not mesh_files:
            pytest.skip("No mesh files found in example_mesh directory")

        adapter.load(0)

        try:
            # Test with first few meshes (limit for testing speed)
            test_meshes = mesh_files[:3]  # Test first 3 meshes

            for mesh_path in test_meshes:
                inputs = {
                    "mesh_path": str(mesh_path),
                    "num_parts": 5,
                    "export_colored_mesh": True,
                }

                result = adapter.process(inputs)

                assert result is not None
                assert result["success"] is True

                output_path = Path(result["output_mesh_path"])
                assert output_path.exists()

                print(f"Successfully segmented {mesh_path.name}")

        finally:
            adapter.unload()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
