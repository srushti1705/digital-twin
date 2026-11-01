"""
Phase 1 Environment Verification Script
----------------------------------------
Verifies that all required climate data processing libraries (numpy, pandas,
matplotlib, seaborn, scikit-learn, xarray, netCDF4, scipy) are installed and
importable, and checks that essential project directory structures exist.
"""

import importlib
import sys
from pathlib import Path
from typing import Tuple

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent


def check_library_imports() -> Tuple[int, int]:
    """
    Tests importability of required Phase 1 libraries and prints status messages.

    Returns:
        tuple (passed_count, total_count)
    """
    libraries = [
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("sklearn", "Scikit-Learn"),
        ("xarray", "xarray"),
        ("netCDF4", "netCDF4"),
        ("scipy", "SciPy"),
    ]

    print("\n1. VERIFYING REQUIRED PYTHON LIBRARIES:")
    print("-" * 55)

    passed = 0
    for module_name, display_name in libraries:
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", "Installed")
            print(f"  [SUCCESS] {display_name:<15} (v{version})")
            passed += 1
        except ImportError as err:
            print(f"  [FAILED]  {display_name:<15} -> Error: {err}")

    return passed, len(libraries)


def verify_project_directories() -> Tuple[int, int]:
    """
    Verifies that all Phase 1 project directories exist on disk.

    Returns:
        tuple (found_count, total_count)
    """
    required_dirs = [
        "data/raw",
        "data/processed",
        "data/external",
        "notebooks",
        "models",
        "outputs",
        "src/data",
        "src/preprocessing",
        "src/models",
        "src/visualization",
        "src/simulation",
    ]

    print("\n2. VERIFYING PROJECT DIRECTORY STRUCTURE:")
    print("-" * 55)

    found = 0
    for dir_rel_path in required_dirs:
        dir_path = BASE_DIR / dir_rel_path
        if dir_path.exists() and dir_path.is_dir():
            print(f"  [EXISTS]  {dir_rel_path:<25}")
            found += 1
        else:
            print(f"  [MISSING] {dir_rel_path:<25} -> Path: {dir_path}")

    return found, len(required_dirs)


def main():
    """Runs the Phase 1 environment verification check."""
    print("=" * 60)
    print("    PHASE 1 CLIMATE DATA ENVIRONMENT VERIFICATION")
    print("=" * 60)

    libs_passed, libs_total = check_library_imports()
    dirs_found, dirs_total = verify_project_directories()

    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY:")
    print(f"  * Libraries:   {libs_passed}/{libs_total} Passed")
    print(f"  * Directories: {dirs_found}/{dirs_total} Verified")

    if libs_passed == libs_total and dirs_found == dirs_total:
        print("\n[SUCCESS] OVERALL STATUS: ALL CHECKS PASSED! Phase 1 environment ready.")
        print("=" * 60)
    else:
        print("\n[WARNING] OVERALL STATUS: VERIFICATION ISSUES DETECTED.")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
