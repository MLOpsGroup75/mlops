#!/usr/bin/env python3
"""
Test runner script for MLOps project

This script provides a convenient way to run pytest with different options.
"""
import subprocess
import sys
import argparse
import os


def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        print(f"{description} - PASSED")
        return True
    else:
        print(f"{description} - FAILED")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run tests for MLOps project")
    parser.add_argument(
        "--coverage", action="store_true", help="Run tests with coverage report"
    )
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--api", action="store_true", help="Run only API tests")
    parser.add_argument(
        "--predict", action="store_true", help="Run only prediction service tests"
    )
    parser.add_argument("--slow", action="store_true", help="Include slow tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet output")
    parser.add_argument("--file", type=str, help="Run tests from specific file")
    parser.add_argument(
        "--test", type=str, help="Run specific test (test file::test function)"
    )
    parser.add_argument(
        "--html-report", action="store_true", help="Generate HTML coverage report"
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install test dependencies before running tests",
    )

    args = parser.parse_args()

    # Change to project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)

    success = True

    # Install dependencies if requested
    if args.install_deps:
        print("Installing test dependencies...")
        install_cmd = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        if not run_command(install_cmd, "Installing dependencies"):
            return 1

    # Build pytest command
    pytest_cmd = [sys.executable, "-m", "pytest"]

    # Add verbosity flags
    if args.verbose:
        pytest_cmd.append("-v")
    elif args.quiet:
        pytest_cmd.append("-q")

    # Add coverage options
    if args.coverage:
        pytest_cmd.extend(
            [
                "--cov=services/api/app",
                "--cov=services/predict/app",
                "--cov=services/common",
                "--cov=config",
            ]
        )
        if args.html_report:
            pytest_cmd.extend(["--cov-report=html", "--cov-report=term"])
        else:
            pytest_cmd.append("--cov-report=term-missing")

    # Add marker-based filtering
    markers = []
    if args.unit:
        markers.append("unit")
    if args.api:
        markers.append("api")
    if args.predict:
        markers.append("predict")

    if markers:
        marker_expr = " or ".join(markers)
        pytest_cmd.extend(["-m", marker_expr])

    # Exclude slow tests unless specifically requested
    if not args.slow:
        if markers:
            marker_expr = f"({' or '.join(markers)}) and not slow"
            pytest_cmd[-1] = marker_expr  # Replace the last -m argument
        else:
            pytest_cmd.extend(["-m", "not slow"])

    # Add specific file or test
    if args.file:
        pytest_cmd.append(f"tests/{args.file}")
    elif args.test:
        pytest_cmd.append(f"tests/{args.test}")
    else:
        # Target both service test directories
        pytest_cmd.extend(["services/api/tests/", "services/predict/tests/"])

    # Run the tests
    if not run_command(pytest_cmd, "Running tests"):
        success = False

    # Summary
    print(f"\n{'='*60}")
    if success:
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        if args.coverage and args.html_report:
            print("📊 Coverage report generated in htmlcov/index.html")
    else:
        print("SOME TESTS FAILED!")
        print("Check the output above for details.")
    print("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
