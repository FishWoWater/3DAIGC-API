"""
Format conversion utilities.

This module provides utilities for converting between different 3D file formats
using Blender as the conversion engine.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def fbx_to_glb(fbx_path: str, output_path: Optional[str] = None) -> str:
    """
    Convert FBX file to GLB format using Blender.

    Args:
        fbx_path: Path to the input FBX file
        output_path: Optional path for the output GLB file. If not provided,
                    will use the same directory and filename as input with .glb extension

    Returns:
        str: Path to the converted GLB file

    Raises:
        ImportError: If bpy (Blender Python module) is not available
        FileNotFoundError: If the input FBX file doesn't exist
        RuntimeError: If conversion fails
    """
    try:
        import bpy
    except ImportError:
        raise ImportError(
            "bpy (Blender Python module) is required for FBX to GLB conversion. "
            "Make sure Blender is installed and bpy is available in your environment."
        )

    # Validate input file
    fbx_path = Path(fbx_path)
    if not fbx_path.exists():
        raise FileNotFoundError(f"Input FBX file not found: {fbx_path}")

    # Determine output path
    if output_path is None:
        output_path = fbx_path.with_suffix(".glb")
    else:
        output_path = Path(output_path)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Converting FBX to GLB: {fbx_path} -> {output_path}")

    try:
        # Clear existing scene
        bpy.ops.wm.read_factory_settings(use_empty=True)

        # Import FBX file
        bpy.ops.import_scene.fbx(filepath=str(fbx_path))

        # Export as GLB
        bpy.ops.export_scene.gltf(
            filepath=str(output_path),
            export_format="GLB",
            use_selection=False,
            export_extras=True,
            export_yup=True,
            export_apply=True,
            export_animations=True,
            export_frame_range=True,
            export_def_bones=True,
            export_optimize_animation_size=False,
            export_anim_single_armature=True,
            export_reset_pose_bones=True,
            export_current_frame=False,
            export_skins=True,
            export_all_influences=False,
            export_morph=True,
            export_morph_normal=True,
            export_morph_tangent=False,
            export_lights=True,
            will_save_settings=False,
        )

        # Verify the output file was created
        if not output_path.exists():
            raise RuntimeError(f"GLB file was not created at: {output_path}")

        logger.info(f"Successfully converted FBX to GLB: {output_path}")
        return str(output_path)

    except Exception as e:
        error_msg = f"Failed to convert FBX to GLB: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def fbx_to_glb_headless(fbx_path: str, output_path: Optional[str] = None) -> str:
    """
    Convert FBX file to GLB format using Blender in headless mode.

    This function runs Blender as a subprocess in headless mode, which is useful
    when running in environments where GUI is not available or when you need
    better isolation.

    Args:
        fbx_path: Path to the input FBX file
        output_path: Optional path for the output GLB file. If not provided,
                    will use the same directory and filename as input with .glb extension

    Returns:
        str: Path to the converted GLB file

    Raises:
        FileNotFoundError: If the input FBX file doesn't exist or Blender is not found
        RuntimeError: If conversion fails
    """
    import subprocess

    # Validate input file
    fbx_path = Path(fbx_path)
    if not fbx_path.exists():
        raise FileNotFoundError(f"Input FBX file not found: {fbx_path}")

    # Determine output path
    if output_path is None:
        output_path = fbx_path.with_suffix(".glb")
    else:
        output_path = Path(output_path)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Converting FBX to GLB (headless): {fbx_path} -> {output_path}")

    # Create temporary Python script for Blender
    script_content = f'''
import bpy

# Clear existing scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Import FBX file
bpy.ops.import_scene.fbx(filepath="{fbx_path}")

# Export as GLB
bpy.ops.export_scene.gltf(
    filepath="{output_path}",
    export_format='GLB',
    use_selection=False,
    export_extras=True,
    export_yup=True,
    export_apply=True,
    export_animations=True,
    export_frame_range=True,
    export_def_bones=True,
    export_optimize_animation_size=False,
    export_anim_single_armature=True,
    export_reset_pose_bones=True,
    export_current_frame=False,
    export_skins=True,
    export_all_influences=False,
    export_morph=True,
    export_morph_normal=True,
    export_morph_tangent=False,
    export_lights=True,
    export_displacement=False,
    will_save_settings=False
)

print("Conversion completed successfully")
'''

    # Write script to temporary file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as script_file:
        script_file.write(script_content)
        script_path = script_file.name

    try:
        # Run Blender in headless mode
        cmd = ["blender", "--background", "--python", script_path]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        if result.returncode != 0:
            error_msg = f"Blender conversion failed: {result.stderr}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Verify the output file was created
        if not output_path.exists():
            raise RuntimeError(f"GLB file was not created at: {output_path}")

        logger.info(f"Successfully converted FBX to GLB (headless): {output_path}")
        return str(output_path)

    except subprocess.TimeoutExpired:
        error_msg = "Blender conversion timed out"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    except FileNotFoundError:
        error_msg = (
            "Blender executable not found. Make sure Blender is installed and in PATH."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    finally:
        # Clean up temporary script file
        try:
            os.unlink(script_path)
        except OSError:
            pass
