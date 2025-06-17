#!/usr/bin/env python3
"""
Test runner for ATTRIEVAL unit tests

This script runs the comprehensive unit tests for ATTRIEVAL and provides
a summary of the test results.
"""

import sys
import os
import subprocess
from pathlib import Path

def main():
    """Run the ATTRIEVAL unit tests."""
    print("🧪 ATTRIEVAL Unit Test Runner")
    print("=" * 50)
    
    # Get the current directory
    current_dir = Path(__file__).parent
    test_file = current_dir / "tests" / "test_attrieval.py"
    
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return 1
    
    print(f"📁 Test file: {test_file}")
    print("🚀 Running ATTRIEVAL unit tests...")
    print()
    
    # Run pytest with verbose output
    cmd = [
        sys.executable, "-m", "pytest", 
        str(test_file),
        "-v",                    # Verbose output
        "--tb=short",           # Short traceback format
        "--color=yes",          # Colored output
        "-x",                   # Stop on first failure
        "--strict-markers",     # Strict marker handling
    ]
    
    try:
        result = subprocess.run(cmd, cwd=current_dir, capture_output=False)
        
        if result.returncode == 0:
            print("\n✅ All ATTRIEVAL tests passed!")
            print("🎉 The ATTRIEVAL implementation is working correctly.")
        else:
            print(f"\n❌ Some tests failed (exit code: {result.returncode})")
            print("🔍 Check the output above for details.")
        
        return result.returncode
        
    except FileNotFoundError:
        print("❌ pytest not found. Please install pytest:")
        print("   pip install pytest")
        return 1
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 