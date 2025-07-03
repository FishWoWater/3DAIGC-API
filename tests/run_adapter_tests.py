#!/usr/bin/env python3
"""
Test runner for all adapter tests.

This script runs tests for all adapters including:
- UniRig (auto-rigging)
- TRELLIS (text-to-mesh)
- PartField (mesh segmentation)
- HoloPart (part completion)
"""

import argparse
import logging
import subprocess
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_pytest_tests():
    """Run the comprehensive pytest tests."""
    logger.info("=== Running Comprehensive pytest Tests ===")

    test_modules = [
        "tests.test_adapters.test_trellis_adapter",
        "tests.test_adapters.test_partfield_adapter",
        "tests.test_adapters.test_holopart_adapter",
        "tests.test_adapters.test_unirig_adapter",  # If it exists
        "tests.test_adapters.test_hunyuan3d_adapter",
        "tests.test_adapters.test_partpacker_adapter",
    ]

    results = {}
    for module in test_modules:
        logger.info(f"Running {module}...")
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    f"{module.replace('.', '/')}.py",
                    "-v",
                    "-s",
                    "--tb=short",
                ],
                capture_output=True,
                text=True,
                timeout=3600,
            )  # 1 hour timeout

            if result.returncode == 0:
                logger.info(f"✓ {module} passed")
                results[module] = "PASSED"
            else:
                logger.error(f"✗ {module} failed")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                results[module] = "FAILED"
        except subprocess.TimeoutExpired:
            logger.error(f"✗ {module} timed out")
            results[module] = "TIMEOUT"
        except Exception as e:
            logger.error(f"✗ {module} error: {e}")
            results[module] = "ERROR"

    return results


def print_summary(simple_results, pytest_results):
    """Print a summary of all test results."""
    logger.info("=== Test Summary ===")

    total_tests = len(simple_results) + len(pytest_results)
    passed_tests = 0

    logger.info("Simple Tests:")
    for test, result in simple_results.items():
        status_emoji = "✓" if result == "PASSED" else "✗"
        logger.info(f"  {status_emoji} {test}: {result}")
        if result == "PASSED":
            passed_tests += 1

    logger.info("Pytest Tests:")
    for test, result in pytest_results.items():
        status_emoji = "✓" if result == "PASSED" else "✗"
        logger.info(f"  {status_emoji} {test}: {result}")
        if result == "PASSED":
            passed_tests += 1

    logger.info(f"Total: {passed_tests}/{total_tests} tests passed")

    return passed_tests == total_tests


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run adapter tests")
    parser.add_argument(
        "--pytest-only",
        action="store_true",
        help="Run only pytest tests",
        default=True,
    )
    parser.add_argument(
        "--adapter",
        choices=[
            "unirig",
            "trellis",
            "partfield",
            "holopart",
            "hunyuan3d",
            "partpacker",
        ],
        help="Run tests for specific adapter only",
    )

    args = parser.parse_args()

    # Change to backend directory
    # backend_dir = Path(__file__).parent
    # original_cwd = Path.cwd()

    # os.chdir(backend_dir)

    simple_results = {}
    pytest_results = {}

    if args.adapter:
        # Run specific adapter pytest
        module_map = {
            "trellis": "tests.test_adapters.test_trellis_adapter",
            "partfield": "tests.test_adapters.test_partfield_adapter",
            "holopart": "tests.test_adapters.test_holopart_adapter",
            "unirig": "tests.test_adapters.test_unirig_adapter",
            "hunyuan3d": "tests.test_adapters.test_hunyuan3d_adapter",
            "partpacker": "tests.test_adapters.test_partpacker_adapter",
        }
        if args.adapter in module_map:
            module = module_map[args.adapter]
            logger.info(f"Running {module} pytest...")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    f"{module.replace('.', '/')}.py",
                    "-v",
                    "-s",
                    "--tb=short",
                ],
                capture_output=True,
                text=True,
                timeout=3600,
            )
            pytest_results = {module: "PASSED" if result.returncode == 0 else "FAILED"}
    else:
        pytest_results = run_pytest_tests()

    success = print_summary(simple_results, pytest_results)

    if success:
        logger.info("🎉 All tests passed!")
        return 0
    else:
        logger.error("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
