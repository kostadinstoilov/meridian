#!/usr/bin/env python3
"""Script to run tests for the meridian-ml-service clustering functionality."""

import subprocess
import sys
import os


def run_command(cmd, description):
    """Run a command and print the results."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("STDOUT:")
    print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    print(f"Return code: {result.returncode}")
    
    return result.returncode == 0


def main():
    """Run the test suite."""
    print("🧪 Running Meridian ML Service Test Suite")
    print("="*60)
    
    # Change to the service directory
    service_dir = "/home/ubuntu/workspace/meridian/services/meridian-ml-service"
    os.chdir(service_dir)
    
    # Check if we're in the right directory
    if not os.path.exists("pyproject.toml"):
        print("❌ Error: pyproject.toml not found. Make sure you're in the service directory.")
        return 1
    
    # Install dependencies if needed
    print("\n📦 Installing dependencies...")
    install_cmd = [sys.executable, "-m", "uv", "pip", "install", "-e", ".[dev]"]
    if not run_command(install_cmd, "Installing test dependencies"):
        print("❌ Failed to install dependencies")
        return 1
    
    # Run different test suites
    test_results = []
    
    # 1. Run unit tests
    test_results.append(run_command([
        sys.executable, "-m", "pytest", 
        "tests/test_clustering_unit.py", 
        "-v", "--cov=src/meridian_ml_service/clustering", 
        "--cov-report=term-missing"
    ], "Unit Tests"))
    
    # 2. Run integration tests
    test_results.append(run_command([
        sys.executable, "-m", "pytest", 
        "tests/test_cluster_endpoint.py", 
        "-v", "--cov=src/meridian_ml_service/main", 
        "--cov-report=term-missing"
    ], "Integration Tests"))
    
    # 3. Run all tests with markers
    test_results.append(run_command([
        sys.executable, "-m", "pytest", 
        "tests/", 
        "-v", 
        "-m", "not slow and not perf"
    ], "All Tests (excluding slow/perf)"))
    
    # 4. Run slow tests locally
    test_results.append(run_command([
        sys.executable, "-m", "pytest", 
        "tests/", 
        "-v", 
        "-m", "slow"
    ], "Slow Tests (local only)"))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())