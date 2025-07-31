#!/usr/bin/env python3
"""
Alternative test runner using pytest.

Provides additional features like coverage reporting and parallel execution.
"""

import subprocess
import sys
import os


def run_pytest():
    """Run tests using pytest."""
    # Add src to Python path
    src_path = os.path.join(os.path.dirname(__file__), 'src')
    env = os.environ.copy()
    
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{src_path}:{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = src_path
    
    # Pytest command with useful options
    cmd = [
        sys.executable, '-m', 'pytest',
        'tests/',                    # Test directory
        '-v',                        # Verbose output
        '--tb=short',               # Shorter traceback format
        '--durations=10',           # Show 10 slowest tests
        '--cov=src',                # Coverage for src directory
        '--cov-report=term-missing', # Show missing lines
        '--cov-report=html',        # Generate HTML coverage report
        '--color=yes'               # Colored output
    ]
    
    print("🧪 Running tests with pytest...")
    print("Command:", ' '.join(cmd))
    print("="*60)
    
    try:
        result = subprocess.run(cmd, env=env)
        return result.returncode == 0
    except FileNotFoundError:
        print("❌ pytest not found. Install with: pip install pytest pytest-cov")
        return False


if __name__ == '__main__':
    success = run_pytest()
    
    if success:
        print("\n🎉 All tests passed!")
        print("📊 Coverage report generated in htmlcov/index.html")
    else:
        print("\n❌ Some tests failed.")
    
    sys.exit(0 if success else 1)
