"""
Run all StegaStamp PyTorch tests.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import test_models
import test_utils


def main():
    print("\n" + "="*60)
    print("StegaStamp PyTorch Test Suite")
    print("="*60 + "\n")

    all_passed = True

    # Run model tests
    if not test_models.run_all_tests():
        all_passed = False

    print()

    # Run utility tests
    if not test_utils.run_all_tests():
        all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("="*60)
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("="*60)
        return 1


if __name__ == '__main__':
    sys.exit(main())
