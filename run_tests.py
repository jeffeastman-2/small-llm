#!/usr/bin/env python3
"""
Test runner for the Small-LLM project.

This script runs all unit tests and provides a summary of results.
"""

import sys
import os
import unittest
import time
from io import StringIO

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def run_tests(test_pattern='test_*.py', verbosity=2):
    """
    Run all tests matching the pattern.
    
    Args:
        test_pattern: Pattern to match test files
        verbosity: Test verbosity level (0-2)
    
    Returns:
        bool: True if all tests passed, False otherwise
    """
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.join(os.path.dirname(__file__), 'tests')
    
    # Discover tests
    suite = loader.discover(start_dir, pattern=test_pattern)
    
    # Count tests
    test_count = suite.countTestCases()
    print(f"🧪 Discovered {test_count} tests")
    print("="*60)
    
    # Run tests
    runner = unittest.TextTestRunner(
        verbosity=verbosity,
        stream=sys.stdout,
        buffer=True  # Capture output from tests
    )
    
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Print summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    print(f"⏱️  Duration: {end_time - start_time:.2f} seconds")
    print(f"🧪 Tests run: {result.testsRun}")
    print(f"✅ Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    
    if result.failures:
        print(f"❌ Failures: {len(result.failures)}")
        
    if result.errors:
        print(f"💥 Errors: {len(result.errors)}")
    
    if result.skipped:
        print(f"⏭️  Skipped: {len(result.skipped)}")
    
    # Print details for failures and errors
    if result.failures:
        print("\n🔴 FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}")
            if verbosity > 1:
                print(f"    {traceback.strip()}")
    
    if result.errors:
        print("\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}")
            if verbosity > 1:
                print(f"    {traceback.strip()}")
    
    # Overall result
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n❌ {len(result.failures) + len(result.errors)} TESTS FAILED")
    
    return success


def run_specific_test(test_name, verbosity=2):
    """
    Run a specific test module.
    
    Args:
        test_name: Name of test module (e.g., 'test_tokenizer')
        verbosity: Test verbosity level
    """
    pattern = f"{test_name}.py"
    return run_tests(pattern, verbosity)


def check_dependencies():
    """Check if required dependencies are available."""
    missing_deps = []
    
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import fitz
    except ImportError:
        missing_deps.append("PyMuPDF")
    
    if missing_deps:
        print("⚠️  Missing dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\nInstall with: pip install -r requirements.txt")
        print("Note: Some tests may be skipped due to missing dependencies.")
        print()


def main():
    """Main test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run tests for Small-LLM project")
    parser.add_argument(
        '--test', '-t', 
        help="Run specific test module (e.g., 'test_tokenizer')",
        default=None
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Verbose output"
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help="Quiet output (minimal)"
    )
    
    args = parser.parse_args()
    
    # Set verbosity
    if args.quiet:
        verbosity = 0
    elif args.verbose:
        verbosity = 2
    else:
        verbosity = 1
    
    print("🧪 Small-LLM Test Runner")
    print("="*60)
    
    # Check dependencies
    check_dependencies()
    
    # Run tests
    if args.test:
        success = run_specific_test(args.test, verbosity)
    else:
        success = run_tests(verbosity=verbosity)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
