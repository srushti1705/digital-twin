"""
Unit test for src.data.test_environment module
------------------------------------------------
Verifies that library import and directory verification functions execute correctly.
"""

from src.data.test_environment import check_library_imports, verify_project_directories


def test_library_imports_pass():
    """Verify that all 8 required Phase 1 libraries pass import check."""
    passed, total = check_library_imports()
    assert passed == total == 8


def test_project_directories_verify():
    """Verify that all 11 required project directories are found."""
    found, total = verify_project_directories()
    assert found == total == 11
