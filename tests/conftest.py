"""
Pytest fixtures for image_hash tests.

Provides image sources (dummy, synthetic similar, from paths).
"""

from pathlib import Path

import pytest

from image_hash.types import NumericArray

from ._helpers import load_images_from_paths, make_dummy_images, make_similar_images

# --- Paths ---


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def default_data_dir(project_root: Path) -> Path:
    return project_root / "data"


@pytest.fixture(scope="session")
def default_image_paths(default_data_dir: Path) -> dict[str, Path]:
    """Default real image paths (source, positive, negative)."""
    return {
        "source": default_data_dir / "source.jpg",
        "positive": default_data_dir / "positive.jpg",
        "negative": default_data_dir / "negative.jpg",
    }


@pytest.fixture(scope="session")
def real_images_available(default_image_paths: dict[str, Path]) -> bool:
    return all(p.exists() for p in default_image_paths.values())


# --- Image data ---


@pytest.fixture
def dummy_images() -> dict[str, NumericArray]:
    """In-memory random BGR images (no file I/O)."""
    return make_dummy_images()


@pytest.fixture
def similar_images() -> dict[str, NumericArray]:
    """Synthetic source, near-duplicate positive, and unrelated negative images."""
    return make_similar_images()


@pytest.fixture(scope="session")
def real_images(
    default_image_paths: dict[str, Path], real_images_available: bool
) -> dict[str, NumericArray]:
    """Real images from default data dir. Skips if files are missing."""
    if not real_images_available:
        pytest.skip(
            "Data images not found (data/source.jpg, data/positive.jpg, data/negative.jpg)"
        )
    return load_images_from_paths(default_image_paths)
