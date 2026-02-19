import sys
import os

def run_tests():
    print("🚀 Starting NeuroCore Test Runner...")

    # Set the working directory to the project root for imports to work
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    # Run pytest programmatically to avoid PATH issues
    try:
        import pytest
    except ImportError:
        print("\n❌ Error: 'pytest' is not installed. Please install it with 'pip install pytest'")
        sys.exit(1)

    # Allow passing command-line arguments to pytest. Default to running all tests verbosely.
    pytest_args = sys.argv[1:]
    if not pytest_args:
        pytest_args = ["tests", "-v"]

    # pytest.main returns an exit code that we can check
    exit_code = pytest.main(pytest_args)

    if exit_code == 0:
        print("\n✅ All tests passed successfully!")
    else:
        print(f"\n❌ Tests failed with exit code {exit_code}")

    sys.exit(exit_code)

if __name__ == "__main__":
    run_tests()
