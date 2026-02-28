import os
import tomllib

from anndata_fcs import __version__


def test_version() -> None:
    """Load version from pyproject.toml and compare it to the version in the package."""
    pyproject_path = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
    assert os.path.exists(pyproject_path), "pyproject.toml not found"
    with open(pyproject_path, "rb") as f:
        pyproject_data = tomllib.load(f)
    expected_version = pyproject_data["project"]["version"]
    assert __version__ == expected_version
