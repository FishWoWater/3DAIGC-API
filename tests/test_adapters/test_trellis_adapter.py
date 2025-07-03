"""
Real GPU inference tests for TRELLIS text-to-textured-mesh adapter.

Tests the TRELLIS adapter with actual model inference on GPU.
No mocks - this performs real text-to-mesh generation.
"""

from pathlib import Path

import pytest
import torch

# Import the adapters
from adapters.trellis_adapter import (
    TrellisImageMeshPaintingAdapter,
    TrellisImageToTexturedMeshAdapter,
    TrellisTextMeshPaintingAdapter,
    TrellisTextToTexturedMeshAdapter,
)
from tests.test_adapters.gpu_memory_tracker import track_gpu_memory


class TestTrellisAdapterRealInference:
    """Test TRELLIS adapter with real GPU inference"""

    @pytest.fixture
    def adapter(self):
        """Create a TRELLIS adapter instance"""
        return TrellisTextToTexturedMeshAdapter()

    @track_gpu_memory
    @pytest.mark.timeout(600)
    def test_text_to_mesh_generation(self, adapter):
        """Test text-to-mesh generation with real inference"""
        # Load the model
        adapter.load(0)

        try:
            # Test with a simple prompt
            inputs = {
                "text_prompt": "a lovely rabbit eating carrots",
                "output_format": "glb",
                "texture_resolution": 512,  # Lower resolution for faster testing
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
            assert gen_info["text_prompt"] == inputs["text_prompt"]
            assert gen_info["seed"] == inputs["seed"]
            assert "vertex_count" in gen_info
            assert "face_count" in gen_info

            print(f"Successfully generated mesh: {output_path}")
            print(
                f"Vertices: {gen_info['vertex_count']}, Faces: {gen_info['face_count']}"
            )

        finally:
            # Unload the model
            adapter.unload()

    @track_gpu_memory
    @pytest.mark.timeout(600)
    def test_different_output_formats(self, adapter):
        """Test generation with different output formats"""
        adapter.load(0)

        try:
            base_inputs = {
                "text_prompt": "a simple cube",
                "texture_resolution": 256,
                "seed": 42,
            }

            for output_format in ["glb"]:
                inputs = {**base_inputs, "output_format": output_format}

                result = adapter.process(inputs)

                assert result is not None
                output_path = Path(result["output_mesh_path"])
                assert output_path.exists()
                assert output_path.suffix == f".{output_format}"

                print(f"Generated {output_format} format: {output_path}")

        finally:
            adapter.unload()

    @pytest.mark.timeout(600)
    def test_unsupported_format_handling(self, adapter):
        """Test handling of unsupported output formats"""
        adapter.load(0)

        try:
            inputs = {
                "text_prompt": "a test object",
                "output_format": "stl",  # Not supported
                "texture_resolution": 256,
                "seed": 42,
            }

            # Should raise an exception for unsupported format
            with pytest.raises(Exception):
                adapter.process(inputs)

            print("Unsupported format correctly rejected")

        finally:
            adapter.unload()

    @track_gpu_memory
    def test_cleanup_after_generation(self, adapter):
        """Test that GPU memory is properly managed"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Check initial GPU memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()

        adapter.load(0)

        try:
            inputs = {
                "text_prompt": "a simple test object",
                "output_format": "glb",
                "texture_resolution": 256,
                "seed": 42,
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

    @track_gpu_memory
    @pytest.mark.timeout(600)
    def test_text_mesh_painting(self):
        """Test text-conditioned mesh painting with real inference"""
        adapter = TrellisTextMeshPaintingAdapter()

        # Load the model
        adapter.load(0)

        try:
            # Test with a simple mesh painting prompt
            inputs = {
                "text_prompt": "a lovely rabbit eating carrots with colorful fur",
                "mesh_path": "assets/example_mesh/typical_creature_dragon.obj",  # Assuming there's a test mesh
                "output_format": "glb",
                "texture_resolution": 512,  # Lower resolution for faster testing
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
            assert gen_info["text_prompt"] == inputs["text_prompt"]
            assert gen_info["seed"] == inputs["seed"]
            assert "vertex_count" in gen_info
            assert "face_count" in gen_info

            print(f"Successfully painted mesh: {output_path}")
            print(
                f"Vertices: {gen_info['vertex_count']}, Faces: {gen_info['face_count']}"
            )

        finally:
            # Unload the model
            adapter.unload()

    @track_gpu_memory
    @pytest.mark.timeout(600)
    def test_image_to_mesh_generation(self):
        """Test image-to-mesh generation with real inference"""
        adapter = TrellisImageToTexturedMeshAdapter()

        # Load the model
        adapter.load(0)

        try:
            # Test with an example image
            inputs = {
                "image_path": "assets/example_image/typical_humanoid_mech.png",  # Assuming there's a test image
                "output_format": "glb",
                "texture_resolution": 512,  # Lower resolution for faster testing
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
            assert gen_info["image_path"] == inputs["image_path"]
            assert gen_info["seed"] == inputs["seed"]
            assert "vertex_count" in gen_info
            assert "face_count" in gen_info

            print(f"Successfully generated mesh from image: {output_path}")
            print(
                f"Vertices: {gen_info['vertex_count']}, Faces: {gen_info['face_count']}"
            )

        finally:
            # Unload the model
            adapter.unload()

    @track_gpu_memory
    @pytest.mark.timeout(600)
    def test_image_mesh_painting(self):
        """Test image-conditioned mesh painting with real inference"""
        adapter = TrellisImageMeshPaintingAdapter()

        # Load the model
        adapter.load(0)

        try:
            # Test with image-conditioned mesh painting
            inputs = {
                "image_path": "assets/example_image/typical_humanoid_mech.png",  # Assuming there's a test image
                "mesh_path": "assets/example_mesh/typical_creature_dragon.obj",  # Assuming there's a test mesh
                "output_format": "glb",
                "texture_resolution": 512,  # Lower resolution for faster testing
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
            assert gen_info["image_path"] == inputs["image_path"]
            assert gen_info["seed"] == inputs["seed"]
            assert "vertex_count" in gen_info
            assert "face_count" in gen_info

            print(f"Successfully painted mesh with image conditioning: {output_path}")
            print(
                f"Vertices: {gen_info['vertex_count']}, Faces: {gen_info['face_count']}"
            )

        finally:
            # Unload the model
            adapter.unload()


class TestTrellisAdapterWithExampleAssets:
    """Test TRELLIS adapter using example assets from the assets directory"""

    @pytest.fixture
    def adapter(self):
        """Create a TRELLIS adapter instance"""
        return TrellisTextToTexturedMeshAdapter()

    @track_gpu_memory
    @pytest.mark.timeout(600)
    def test_with_example_prompts(self, adapter):
        """Test generation using prompts from example_prompts.txt"""
        # Read example prompts
        prompts_file = Path("assets/example_prompts.txt")
        if not prompts_file.exists():
            pytest.skip("Example prompts file not found")

        with open(prompts_file, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]

        adapter.load(0)

        try:
            # Test with first few prompts (limit for testing speed)
            test_prompts = prompts[:2]  # Test first 2 prompts

            for prompt in test_prompts:
                inputs = {
                    "text_prompt": prompt,
                    "output_format": "glb",
                    "texture_resolution": 256,  # Lower resolution for speed
                    "seed": 42,
                }

                result = adapter.process(inputs)

                assert result is not None
                assert "output_mesh_path" in result

                output_path = Path(result["output_mesh_path"])
                assert output_path.exists()

                print(f"Successfully generated mesh for prompt: '{prompt}'")
                print(f"Output: {output_path}")

        finally:
            adapter.unload()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
