#!/usr/bin/env python3
"""
Comprehensive Test Client for 3D Generative Models API

This script tests all available models and features using example assets from the assets directory.
It submits jobs, monitors their progress, downloads results, and saves them to organized directories.

Features tested:
- Text-to-textured-mesh (TRELLIS)
- Text-mesh-painting (TRELLIS)
- Image-to-textured-mesh (TRELLIS, Hunyuan3D)
- Image-mesh-painting (TRELLIS, Hunyuan3D)
- Image-to-raw-mesh (Hunyuan3D, PartPacker)
- Mesh segmentation (PartField)
- Part generation/completion (HoloPart)
- Auto-rigging (UniRig)

Execution Modes:
- Concurrent (default): Submit all jobs first, then monitor all jobs until completion
- Sequential: Submit and wait for each job one by one

Usage:
    # Concurrent mode (default)
    python tests/run_test_client.py --server-url http://localhost:7842

    # Sequential mode
    python tests/run_test_client.py --server-url http://localhost:7842 --sequential
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


class ComprehensiveModelTester:
    """Comprehensive test client for all 3D generative models"""

    def __init__(
        self,
        server_url: str = "http://localhost:7842",
        timeout_seconds: int = 600,  # 10 minutes per job
        poll_interval: int = 10,  # Check every 10 seconds
        output_base_dir: str = "test_results",
        concurrent_mode: bool = True,  # Whether to run jobs concurrently or sequentially
    ):
        self.server_url = server_url.rstrip("/")
        self.api_base_url = f"{self.server_url}/api/v1"
        self.timeout_seconds = timeout_seconds
        self.poll_interval = poll_interval
        self.output_base_dir = Path(output_base_dir)
        self.concurrent_mode = concurrent_mode

        # Create output directory structure
        self.setup_output_directories()

        # Setup logging
        self.setup_logging()

        # Test configurations for each feature/model combination
        self.test_configs = self.setup_test_configurations()

        # Track test results
        self.test_results = {
            "summary": {
                "total_tests": 0,
                "successful": 0,
                "failed": 0,
                "skipped": 0,
                "start_time": None,
                "end_time": None,
            },
            "tests": [],
        }

    def setup_output_directories(self):
        """Create organized output directory structure"""
        self.output_base_dir.mkdir(exist_ok=True)

        # Create directories for each feature
        features = [
            "text_to_textured_mesh",
            "text_mesh_painting",
            "image_to_textured_mesh",
            "image_mesh_painting",
            "image_to_raw_mesh",
            "mesh_segmentation",
            "part_completion",
            "auto_rig",
        ]

        for feature in features:
            (self.output_base_dir / feature).mkdir(exist_ok=True)

        # Create logs directory
        (self.output_base_dir / "logs").mkdir(exist_ok=True)

    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_base_dir / "logs" / "test_client.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def setup_test_configurations(self) -> List[Dict[str, Any]]:
        """Setup test configurations for all model/feature combinations"""

        # Read example prompts
        try:
            with open("assets/example_prompts.txt", "r", encoding="utf-8") as f:
                prompts = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            prompts = [
                "A red sports car",
                "A green plant in a pot",
                "A cute rabbit",
            ]

        configs = []

        # 1. Text-to-textured-mesh tests
        configs.extend(
            [
                {
                    "test_name": "trellis_text_to_textured_mesh_english",
                    "feature": "text_to_textured_mesh",
                    "endpoint": "/mesh-generation/text-to-textured-mesh",
                    "model_preference": "trellis_text_to_textured_mesh",
                    "input_data": {
                        "text_prompt": prompts[-2]
                        if len(prompts) >= 2
                        else "A red sports car",
                        "texture_prompt": "realistic metallic texture",
                        "texture_resolution": 1024,
                        "output_format": "glb",
                    },
                    "expected_outputs": ["output_mesh_path"],
                },
                {
                    "test_name": "trellis_text_to_textured_mesh_chinese",
                    "feature": "text_to_textured_mesh",
                    "endpoint": "/mesh-generation/text-to-textured-mesh",
                    "model_preference": "trellis_text_to_textured_mesh",
                    "input_data": {
                        "text_prompt": prompts[0] if prompts else "一片绿色的树叶",
                        "texture_prompt": "自然纹理",
                        "texture_resolution": 1024,
                        "output_format": "glb",
                    },
                    "expected_outputs": ["output_mesh_path"],
                },
            ]
        )

        # 2. Text-mesh-painting tests
        mesh_files = list(Path("assets/example_mesh").glob("*.obj"))[:2]
        for i, mesh_file in enumerate(mesh_files):
            configs.append(
                {
                    "test_name": f"trellis_text_mesh_painting_{mesh_file.stem}",
                    "feature": "text_mesh_painting",
                    "endpoint": "/mesh-generation/text-mesh-painting",
                    "model_preference": "trellis_text_mesh_painting",
                    "input_data": {
                        "text_prompt": "rusty metal texture with weathered surface",
                        "mesh_path": str(mesh_file),
                        "texture_resolution": 1024,
                        "output_format": "glb",
                    },
                    "expected_outputs": ["output_mesh_path"],
                }
            )

        # 3. Image-to-textured-mesh tests
        image_files = list(Path("assets/example_image").glob("*.png"))[:2]

        # TRELLIS models
        for i, image_file in enumerate(image_files):
            configs.append(
                {
                    "test_name": f"trellis_image_to_textured_mesh_{image_file.stem}",
                    "feature": "image_to_textured_mesh",
                    "endpoint": "/mesh-generation/image-to-textured-mesh",
                    "model_preference": "trellis_image_to_textured_mesh",
                    "input_data": {
                        "image_path": str(image_file),
                        "texture_resolution": 1024,
                        "output_format": "glb",
                    },
                    "expected_outputs": ["output_mesh_path"],
                }
            )

        # Hunyuan3D models (just test one image each)
        if image_files:
            configs.extend(
                [
                    {
                        "test_name": f"hunyuan3d_2_0_image_to_textured_mesh_{image_files[0].stem}",
                        "feature": "image_to_textured_mesh",
                        "endpoint": "/mesh-generation/image-to-textured-mesh",
                        "model_preference": "hunyuan3d_2_0_image_to_textured_mesh",
                        "input_data": {
                            "image_path": str(image_files[0]),
                            "texture_resolution": 1024,
                            "output_format": "glb",
                        },
                        "expected_outputs": ["output_mesh_path"],
                    },
                    {
                        "test_name": f"hunyuan3d_image_to_textured_mesh_{image_files[0].stem}",
                        "feature": "image_to_textured_mesh",
                        "endpoint": "/mesh-generation/image-to-textured-mesh",
                        "model_preference": "hunyuan3d_image_to_textured_mesh",
                        "input_data": {
                            "image_path": str(image_files[0]),
                            "texture_resolution": 1024,
                            "output_format": "glb",
                        },
                        "expected_outputs": ["output_mesh_path"],
                    },
                ]
            )

        # 4. Image-mesh-painting tests
        for i, (image_file, mesh_file) in enumerate(
            zip(image_files[:1], mesh_files[:1])
        ):
            configs.extend(
                [
                    {
                        "test_name": f"trellis_image_mesh_painting_{image_file.stem}_{mesh_file.stem}",
                        "feature": "image_mesh_painting",
                        "endpoint": "/mesh-generation/image-mesh-painting",
                        "model_preference": "trellis_image_mesh_painting",
                        "input_data": {
                            "image_path": str(image_file),
                            "mesh_path": str(mesh_file),
                            "texture_resolution": 1024,
                            "output_format": "glb",
                        },
                        "expected_outputs": ["output_mesh_path"],
                    },
                    {
                        "test_name": f"hunyuan3d_image_mesh_painting_{image_file.stem}_{mesh_file.stem}",
                        "feature": "image_mesh_painting",
                        "endpoint": "/mesh-generation/image-mesh-painting",
                        "model_preference": "hunyuan3d_image_mesh_painting",
                        "input_data": {
                            "image_path": str(image_file),
                            "mesh_path": str(mesh_file),
                            "texture_resolution": 1024,
                            "output_format": "glb",
                        },
                        "expected_outputs": ["output_mesh_path"],
                    },
                ]
            )

        # 5. Image-to-raw-mesh tests

        # Hunyuan3D models
        if image_files:
            configs.extend(
                [
                    {
                        "test_name": f"hunyuan3d_2_0_image_to_raw_mesh_{image_files[0].stem}",
                        "feature": "image_to_raw_mesh",
                        "endpoint": "/mesh-generation/image-to-raw-mesh",
                        "model_preference": "hunyuan3d_2_0_image_to_raw_mesh",
                        "input_data": {
                            "image_path": str(image_files[0]),
                            "output_format": "glb",
                        },
                        "expected_outputs": ["output_mesh_path"],
                    },
                    {
                        "test_name": f"hunyuan3d_image_to_raw_mesh_{image_files[0].stem}",
                        "feature": "image_to_raw_mesh",
                        "endpoint": "/mesh-generation/image-to-raw-mesh",
                        "model_preference": "hunyuan3d_image_to_raw_mesh",
                        "input_data": {
                            "image_path": str(image_files[0]),
                            "output_format": "glb",
                        },
                        "expected_outputs": ["output_mesh_path"],
                    },
                ]
            )

        # PartPacker models (uses image_to_raw_mesh endpoint)
        partpacker_images = list(Path("assets/example_partpacker").glob("*.png"))[:2]
        for image_file in partpacker_images:
            configs.append(
                {
                    "test_name": f"partpacker_part_packing_{image_file.stem}",
                    "feature": "image_to_raw_mesh",  # PartPacker uses this endpoint
                    "endpoint": "/mesh-generation/image-to-raw-mesh",
                    "model_preference": "partpacker_part_packing",
                    "input_data": {
                        "image_path": str(image_file),
                        "output_format": "glb",
                    },
                    "expected_outputs": ["output_mesh_path"],
                }
            )

        # 6. Mesh segmentation tests
        seg_meshes = list(Path("assets/example_meshseg").glob("*.glb"))[:2]
        for mesh_file in seg_meshes:
            configs.append(
                {
                    "test_name": f"partfield_mesh_segmentation_{mesh_file.stem}",
                    "feature": "mesh_segmentation",
                    "endpoint": "/mesh-segmentation/segment-mesh",
                    "model_preference": "partfield_mesh_segmentation",
                    "input_data": {
                        "mesh_path": str(mesh_file),
                        "num_parts": 8,
                        "output_format": "glb",
                    },
                    "expected_outputs": ["output_mesh_path"],
                }
            )

        # 7. Part generation/completion tests (HoloPart)
        holopart_meshes = list(Path("assets/example_holopart").glob("*.glb"))[:2]
        for mesh_file in holopart_meshes:
            configs.append(
                {
                    "test_name": f"holopart_part_completion_{mesh_file.stem}",
                    "feature": "part_completion",
                    "endpoint": "/mesh-generation/part-completion",  # Uses part-completion endpoint
                    "model_preference": "holopart_part_completion",
                    "input_data": {
                        "mesh_path": str(mesh_file),
                        "output_format": "glb",
                    },
                    "expected_outputs": ["output_mesh_path"],
                }
            )

        # 8. Auto-rigging tests
        autorig_meshes = list(Path("assets/example_autorig").glob("*.glb"))[:2]
        for mesh_file in autorig_meshes:
            configs.append(
                {
                    "test_name": f"unirig_auto_rig_{mesh_file.stem}",
                    "feature": "auto_rig",
                    "endpoint": "/auto-rig/generate-rig",
                    "model_preference": "unirig_auto_rig",
                    "input_data": {
                        "mesh_path": str(mesh_file),
                        "rig_mode": "skeleton",
                        "output_format": "fbx",
                    },
                    "expected_outputs": ["output_mesh_path"],
                }
            )

        return configs

    def check_server_health(self) -> bool:
        """Check if server is running and healthy"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=10)
            if response.status_code == 200:
                self.logger.info("Server health check passed")
                return True
            else:
                self.logger.error(f"Server health check failed: {response.status_code}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to connect to server: {e}")
            return False

    def get_available_models(self) -> Dict[str, Any]:
        """Get list of available models from server"""
        try:
            response = requests.get(f"{self.api_base_url}/system/features", timeout=30)
            if response.status_code == 200:
                data = response.json()
                self.logger.info(
                    f"Available features retrieved: {len(data.get('features', {}))} features"
                )
                return data
            else:
                self.logger.warning(
                    f"Failed to get available models: {response.status_code}"
                )
                return {}
        except Exception as e:
            self.logger.warning(f"Failed to get available models: {e}")
            return {}

    def submit_job(self, config: Dict[str, Any]) -> Optional[str]:
        """Submit a job to the API"""
        try:
            endpoint = f"{self.api_base_url}{config['endpoint']}"
            data = config["input_data"].copy()
            data["model_preference"] = config["model_preference"]

            response = requests.post(endpoint, json=data, timeout=60)
            if response.status_code == 200:
                result = response.json()
                job_id = result.get("job_id")
                if job_id:
                    self.logger.info(
                        f"Job submitted successfully: {job_id} for {config['test_name']}"
                    )
                    return job_id
                else:
                    self.logger.error(
                        f"No job_id in response for {config['test_name']}"
                    )
                    return None
            else:
                self.logger.error(
                    f"Job submission failed for {config['test_name']}: {response.status_code} - {response.text}"
                )
                return None
        except Exception as e:
            self.logger.error(
                f"Exception during job submission for {config['test_name']}: {e}"
            )
            return None

    def wait_for_job_completion(
        self, job_id: str, test_name: str
    ) -> Optional[Dict[str, Any]]:
        """Wait for job completion and return final status"""
        start_time = time.time()

        while time.time() - start_time < self.timeout_seconds:
            try:
                response = requests.get(
                    f"{self.api_base_url}/system/jobs/{job_id}", timeout=30
                )
                if response.status_code == 200:
                    job_status = response.json()
                    status = job_status.get("status")

                    if status == "completed":
                        elapsed = time.time() - start_time
                        self.logger.info(
                            f"Job {job_id} ({test_name}) completed in {elapsed:.1f}s"
                        )
                        return job_status
                    elif status == "error":
                        error_msg = job_status.get("error", "Unknown error")
                        self.logger.error(
                            f"Job {job_id} ({test_name}) failed: {error_msg}"
                        )
                        return job_status
                    elif status in ["queued", "processing"]:
                        elapsed = time.time() - start_time
                        self.logger.info(
                            f"Job {job_id} ({test_name}) status: {status} (elapsed: {elapsed:.1f}s)"
                        )
                        time.sleep(self.poll_interval)
                        continue
                    elif status in ["failed", "error"]:
                        error_msg = job_status.get("error", "Unknown error")
                        self.logger.error(
                            f"Job {job_id} ({test_name}) failed: {error_msg}"
                        )
                        return job_status
                    else:
                        self.logger.warning(
                            f"Job {job_id} ({test_name}) unexpected status: {status}"
                        )
                        time.sleep(self.poll_interval)
                        continue
                else:
                    self.logger.error(
                        f"Failed to get job status for {job_id}: {response.status_code}"
                    )
                    time.sleep(self.poll_interval)
                    continue
            except Exception as e:
                self.logger.error(f"Exception checking job status for {job_id}: {e}")
                time.sleep(self.poll_interval)
                continue

        self.logger.error(
            f"Job {job_id} ({test_name}) timed out after {self.timeout_seconds}s"
        )
        return None

    def download_result(
        self, job_id: str, test_name: str, feature: str
    ) -> Optional[str]:
        """Download job result file"""
        try:
            # First get job info to check if result exists
            response = requests.get(
                f"{self.api_base_url}/system/jobs/{job_id}/info", timeout=30
            )
            if response.status_code != 200:
                self.logger.error(f"Failed to get job info for {job_id}")
                return None

            job_info = response.json()
            if not job_info.get("file_info", {}).get("file_exists", False):
                self.logger.error(f"No result file exists for job {job_id}")
                return None

            original_filename = job_info["file_info"]["filename"]
            file_size_mb = job_info["file_info"]["file_size_mb"]

            self.logger.info(
                f"Downloading result for {job_id}: {original_filename} ({file_size_mb:.1f} MB)"
            )

            # Download the file
            response = requests.get(
                f"{self.api_base_url}/system/jobs/{job_id}/download", timeout=120
            )
            if response.status_code == 200:
                # Create output filename
                output_dir = self.output_base_dir / feature
                output_file = output_dir / f"{test_name}_{original_filename}"

                # Write file
                with open(output_file, "wb") as f:
                    f.write(response.content)

                self.logger.info(f"Downloaded result to: {output_file}")
                return str(output_file)
            else:
                self.logger.error(
                    f"Failed to download result for {job_id}: {response.status_code} - {response.text}"
                )
                return None

        except Exception as e:
            self.logger.error(f"Exception downloading result for {job_id}: {e}")
            return None

    def run_single_test(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test configuration"""
        test_name = config["test_name"]
        start_time = time.time()

        result = {
            "test_name": test_name,
            "feature": config["feature"],
            "model_preference": config["model_preference"],
            "endpoint": config["endpoint"],
            "status": "unknown",
            "job_id": None,
            "error": None,
            "start_time": start_time,
            "end_time": None,
            "duration": None,
            "downloaded_file": None,
            "job_result": None,
        }

        try:
            self.logger.info(f"Starting test: {test_name}")

            # Submit job
            job_id = self.submit_job(config)
            if not job_id:
                result["status"] = "submission_failed"
                result["error"] = "Failed to submit job"
                return result

            result["job_id"] = job_id

            # Wait for completion
            job_status = self.wait_for_job_completion(job_id, test_name)
            if not job_status:
                result["status"] = "timeout"
                result["error"] = "Job timed out"
                return result

            result["job_result"] = job_status

            if job_status.get("status") == "completed":
                # Download result
                downloaded_file = self.download_result(
                    job_id, test_name, config["feature"]
                )
                if downloaded_file:
                    result["downloaded_file"] = downloaded_file
                    result["status"] = "success"
                else:
                    result["status"] = "download_failed"
                    result["error"] = "Failed to download result"
            else:
                result["status"] = "job_failed"
                result["error"] = job_status.get("error", "Job failed")

        except Exception as e:
            result["status"] = "exception"
            result["error"] = str(e)
            self.logger.error(f"Exception in test {test_name}: {e}")

        finally:
            result["end_time"] = time.time()
            result["duration"] = result["end_time"] - result["start_time"]

        return result

    def submit_all_jobs(self, configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Submit all jobs and return job tracking info"""
        job_tracking = []

        self.logger.info(f"Submitting {len(configs)} jobs...")

        for config in configs:
            test_name = config["test_name"]
            start_time = time.time()

            job_info = {
                "config": config,
                "test_name": test_name,
                "feature": config["feature"],
                "model_preference": config["model_preference"],
                "endpoint": config["endpoint"],
                "status": "submitting",
                "job_id": None,
                "error": None,
                "start_time": start_time,
                "end_time": None,
                "duration": None,
                "downloaded_file": None,
                "job_result": None,
                "submission_time": None,
            }

            try:
                self.logger.info(f"Submitting job: {test_name}")
                job_id = self.submit_job(config)

                if job_id:
                    job_info["job_id"] = job_id
                    job_info["status"] = "submitted"
                    job_info["submission_time"] = time.time()
                    self.logger.info(
                        f"Successfully submitted job {job_id} for {test_name}"
                    )
                else:
                    job_info["status"] = "submission_failed"
                    job_info["error"] = "Failed to submit job"
                    self.logger.error(f"Failed to submit job for {test_name}")

            except Exception as e:
                job_info["status"] = "submission_failed"
                job_info["error"] = str(e)
                self.logger.error(f"Exception submitting job for {test_name}: {e}")

            job_tracking.append(job_info)

            # Brief pause between submissions to avoid overwhelming server
            time.sleep(0.5)

        successful_submissions = len(
            [j for j in job_tracking if j["status"] == "submitted"]
        )
        self.logger.info(
            f"Successfully submitted {successful_submissions}/{len(configs)} jobs"
        )

        return job_tracking

    def monitor_all_jobs(
        self, job_tracking: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Monitor all jobs until completion and download results"""
        active_jobs = [job for job in job_tracking if job["job_id"]]
        completed_jobs = [job for job in job_tracking if not job["job_id"]]

        self.logger.info(f"Monitoring {len(active_jobs)} active jobs...")

        while active_jobs:
            jobs_to_check = active_jobs.copy()
            newly_completed = []

            for job_info in jobs_to_check:
                job_id = job_info["job_id"]
                test_name = job_info["test_name"]

                try:
                    # Check job status
                    response = requests.get(
                        f"{self.api_base_url}/system/jobs/{job_id}", timeout=30
                    )

                    if response.status_code == 200:
                        job_status = response.json()
                        status = job_status.get("status")

                        if status == "completed":
                            job_info["status"] = "completed"
                            job_info["job_result"] = job_status
                            job_info["end_time"] = time.time()
                            job_info["duration"] = (
                                job_info["end_time"] - job_info["start_time"]
                            )

                            # Download result
                            try:
                                downloaded_file = self.download_result(
                                    job_id, test_name, job_info["feature"]
                                )
                                if downloaded_file:
                                    job_info["downloaded_file"] = downloaded_file
                                    job_info["status"] = "success"
                                    self.logger.info(
                                        f"Job {job_id} ({test_name}) completed successfully"
                                    )
                                else:
                                    job_info["status"] = "download_failed"
                                    job_info["error"] = "Failed to download result"
                                    self.logger.error(
                                        f"Job {job_id} ({test_name}) download failed"
                                    )
                            except Exception as e:
                                job_info["status"] = "download_failed"
                                job_info["error"] = f"Download exception: {str(e)}"
                                self.logger.error(
                                    f"Job {job_id} ({test_name}) download exception: {e}"
                                )

                            newly_completed.append(job_info)

                        elif status in ["error", "failed"]:
                            job_info["status"] = "job_failed"
                            job_info["job_result"] = job_status
                            job_info["error"] = job_status.get("error", "Job failed")
                            job_info["end_time"] = time.time()
                            job_info["duration"] = (
                                job_info["end_time"] - job_info["start_time"]
                            )

                            self.logger.error(
                                f"Job {job_id} ({test_name}) failed: {job_info['error']}"
                            )
                            newly_completed.append(job_info)

                        elif status in ["queued", "processing"]:
                            # Job still running, check for timeout
                            elapsed = time.time() - job_info["start_time"]
                            if elapsed > self.timeout_seconds:
                                job_info["status"] = "timeout"
                                job_info["error"] = (
                                    f"Job timed out after {self.timeout_seconds}s"
                                )
                                job_info["end_time"] = time.time()
                                job_info["duration"] = (
                                    job_info["end_time"] - job_info["start_time"]
                                )

                                self.logger.error(
                                    f"Job {job_id} ({test_name}) timed out"
                                )
                                newly_completed.append(job_info)
                            # else: continue monitoring

                        else:
                            self.logger.warning(
                                f"Job {job_id} ({test_name}) unexpected status: {status}"
                            )

                    else:
                        self.logger.warning(
                            f"Failed to get status for job {job_id}: {response.status_code}"
                        )

                except Exception as e:
                    self.logger.error(
                        f"Exception checking job {job_id} ({test_name}): {e}"
                    )
                    # Continue monitoring other jobs

            # Move completed jobs
            for job_info in newly_completed:
                if job_info in active_jobs:
                    active_jobs.remove(job_info)
                completed_jobs.append(job_info)

            if active_jobs:
                # Log progress
                total_jobs = len(job_tracking)
                completed_count = len(completed_jobs)
                active_count = len(active_jobs)

                self.logger.info(
                    f"Progress: {completed_count}/{total_jobs} completed, {active_count} still running"
                )

                # Show status of active jobs
                for job_info in active_jobs[:5]:  # Show first 5 active jobs
                    elapsed = time.time() - job_info["start_time"]
                    self.logger.info(
                        f"  - {job_info['test_name']}: running for {elapsed:.1f}s"
                    )

                if len(active_jobs) > 5:
                    self.logger.info(f"  - ... and {len(active_jobs) - 5} more")

                # Wait before next check
                time.sleep(self.poll_interval)

        self.logger.info("All jobs completed!")
        return completed_jobs

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test configurations in either concurrent or sequential mode"""
        mode_text = "concurrent" if self.concurrent_mode else "sequential"
        self.logger.info(f"Starting comprehensive model testing in {mode_text} mode")
        self.test_results["summary"]["start_time"] = time.time()

        # Check server health
        if not self.check_server_health():
            self.logger.error("Server health check failed, aborting tests")
            return self.test_results

        # Get available models
        available_models = self.get_available_models()
        self.logger.info(
            f"Server has {len(available_models.get('features', {}))} features available"
        )

        # Filter configs to only test available models
        valid_configs = self.test_configs

        self.test_results["summary"]["total_tests"] = len(valid_configs)
        self.logger.info(f"Running {len(valid_configs)} tests in {mode_text} mode")

        if self.concurrent_mode:
            # Concurrent mode: Submit all jobs, then monitor all
            self.logger.info(
                "Using concurrent mode: submitting all jobs first, then monitoring"
            )

            # Step 1: Submit all jobs
            job_tracking = self.submit_all_jobs(valid_configs)

            # Step 2: Monitor all jobs until completion
            completed_jobs = self.monitor_all_jobs(job_tracking)

            # Step 3: Process results
            for job_info in completed_jobs:
                # Convert job_info to the expected test result format
                result = {
                    "test_name": job_info["test_name"],
                    "feature": job_info["feature"],
                    "model_preference": job_info["model_preference"],
                    "endpoint": job_info["endpoint"],
                    "status": job_info["status"],
                    "job_id": job_info["job_id"],
                    "error": job_info["error"],
                    "start_time": job_info["start_time"],
                    "end_time": job_info["end_time"],
                    "duration": job_info["duration"],
                    "downloaded_file": job_info["downloaded_file"],
                    "job_result": job_info["job_result"],
                }

                self.test_results["tests"].append(result)

                if result["status"] == "success":
                    self.test_results["summary"]["successful"] += 1
                elif result["status"] in [
                    "submission_failed",
                    "timeout",
                    "job_failed",
                    "download_failed",
                    "exception",
                ]:
                    self.test_results["summary"]["failed"] += 1
                else:
                    self.test_results["summary"]["skipped"] += 1
        else:
            # Sequential mode: Run tests one by one
            self.logger.info("Using sequential mode: running tests one by one")

            for config in valid_configs:
                result = self.run_single_test(config)
                self.test_results["tests"].append(result)

                if result["status"] == "success":
                    self.test_results["summary"]["successful"] += 1
                elif result["status"] in [
                    "submission_failed",
                    "timeout",
                    "job_failed",
                    "download_failed",
                    "exception",
                ]:
                    self.test_results["summary"]["failed"] += 1
                else:
                    self.test_results["summary"]["skipped"] += 1

                # Brief pause between tests
                time.sleep(2)

        self.test_results["summary"]["end_time"] = time.time()
        total_duration = (
            self.test_results["summary"]["end_time"]
            - self.test_results["summary"]["start_time"]
        )

        self.logger.info("=" * 80)
        self.logger.info("COMPREHENSIVE TEST SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"Total tests: {self.test_results['summary']['total_tests']}")
        self.logger.info(f"Successful: {self.test_results['summary']['successful']}")
        self.logger.info(f"Failed: {self.test_results['summary']['failed']}")
        self.logger.info(f"Skipped: {self.test_results['summary']['skipped']}")
        self.logger.info(f"Total duration: {total_duration:.1f} seconds")
        self.logger.info(f"Execution mode: {mode_text}")
        self.logger.info("=" * 80)

        # Save detailed results
        results_file = self.output_base_dir / "test_results.json"
        with open(results_file, "w") as f:
            json.dump(self.test_results, f, indent=2, default=str)

        self.logger.info(f"Detailed results saved to: {results_file}")

        return self.test_results

    def print_summary_report(self):
        """Print a human-readable summary report"""
        summary = self.test_results["summary"]

        print("\n" + "=" * 100)
        print("COMPREHENSIVE 3D GENERATIVE MODELS TEST REPORT")
        print("=" * 100)

        print("\nOVERALL RESULTS:")
        print(f"  Total Tests:    {summary['total_tests']}")
        print(
            f"  Successful:     {summary['successful']} ({summary['successful'] / summary['total_tests'] * 100:.1f}%)"
        )
        print(
            f"  Failed:         {summary['failed']} ({summary['failed'] / summary['total_tests'] * 100:.1f}%)"
        )
        print(
            f"  Skipped:        {summary['skipped']} ({summary['skipped'] / summary['total_tests'] * 100:.1f}%)"
        )
        print(
            f"  Duration:       {summary['end_time'] - summary['start_time']:.1f} seconds"
        )

        # Group results by feature
        by_feature = {}
        for test in self.test_results["tests"]:
            feature = test.get("feature", "unknown")
            if feature not in by_feature:
                by_feature[feature] = {"success": 0, "failed": 0, "total": 0}
            by_feature[feature]["total"] += 1
            if test.get("status") == "success":
                by_feature[feature]["success"] += 1
            else:
                by_feature[feature]["failed"] += 1

        print("\nRESULTS BY FEATURE:")
        for feature, stats in by_feature.items():
            success_rate = (
                stats["success"] / stats["total"] * 100 if stats["total"] > 0 else 0
            )
            print(
                f"  {feature:25} {stats['success']:2d}/{stats['total']:2d} ({success_rate:5.1f}%)"
            )

        print("\nDETAILED RESULTS:")
        for test in self.test_results["tests"]:
            status = test.get("status", "unknown")
            duration = test.get("duration", 0)
            status_icon = "✓" if status == "success" else "✗"
            print(
                f"  {status_icon} {test.get('test_name', 'unknown'):50} {status:15} {duration:6.1f}s"
            )
            if test.get("error"):
                print(f"    Error: {test['error']}")
            if test.get("downloaded_file"):
                print(f"    Output: {test['downloaded_file']}")

        print(f"\nOUTPUT DIRECTORY: {self.output_base_dir}")
        print("=" * 100)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Comprehensive 3D Generative Models Test Client"
    )
    parser.add_argument(
        "--server-url", default="http://localhost:7842", help="Server URL"
    )
    parser.add_argument(
        "--timeout", type=int, default=900, help="Timeout per job in seconds"
    )
    parser.add_argument(
        "--poll-interval", type=int, default=10, help="Poll interval in seconds"
    )
    parser.add_argument("--output-dir", default="test_results", help="Output directory")
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run tests sequentially instead of concurrently (default: concurrent)",
    )

    args = parser.parse_args()

    # Create and run tester
    tester = ComprehensiveModelTester(
        server_url=args.server_url,
        timeout_seconds=args.timeout,
        poll_interval=args.poll_interval,
        output_base_dir=args.output_dir,
        concurrent_mode=not args.sequential,  # Default to concurrent unless --sequential is specified
    )

    # Run all tests
    results = tester.run_all_tests()

    # Print summary
    tester.print_summary_report()

    return results


if __name__ == "__main__":
    main()
