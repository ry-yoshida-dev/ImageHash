"""
Shared helpers for image_hash tests (no pytest dependency).

Used by test.py (script mode) and conftest.py (fixtures).
"""

from pathlib import Path

import cv2
import numpy as np

from image_hash.hasher import ImageHasher
from image_hash.method import HashMethod
from image_hash.types import FloatArray, NumericArray
from image_hash.wavelet import WaveletHash


def build_hasher(method: HashMethod) -> ImageHasher:
    """Build ImageHasher for the given method (single place for WaveletHash handling)."""
    if method == HashMethod.WAVELET:
        wavelet_obj = method.object
        if not isinstance(wavelet_obj, WaveletHash):
            raise TypeError("HashMethod.WAVELET must provide WaveletHash object.")
        return ImageHasher.build(method=method, wavelet_obj=wavelet_obj)
    return ImageHasher.build(method=method)


def run_cross_distance(
    hasher: ImageHasher,
    images: dict[str, NumericArray],
    method: HashMethod,
) -> FloatArray:
    """
    Compute hashes for all images and return the cross distance matrix.

    Handles bit-convertible vs vector methods internally.
    """
    raw = np.array([hasher.compute(img) for img in images.values()])
    if method.is_bit_convertible:
        binary = hasher.to_binary(raw)
        return hasher.measure_cross_distance(binary, binary)
    return hasher.measure_cross_distance(raw, raw)


def make_dummy_images() -> dict[str, NumericArray]:
    """In-memory random BGR images (source, positive, negative) with no relation to each other."""
    h, w = 64, 64
    rng = np.random.default_rng(114514)
    return {
        "source": rng.integers(0, 256, (h, w, 3), dtype=np.uint8),
        "positive": rng.integers(0, 256, (h, w, 3), dtype=np.uint8),
        "negative": rng.integers(0, 256, (h, w, 3), dtype=np.uint8),
    }


def _make_smooth_image(rng: np.random.Generator) -> NumericArray:
    coarse = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    return cv2.resize(coarse, (128, 128), interpolation=cv2.INTER_CUBIC)


def make_similar_images() -> dict[str, NumericArray]:
    """
    Synthetic BGR images where "positive" is a noisy copy of "source" and "negative" is unrelated.

    Stands in for the real images in data/ when they are not present.
    """
    rng = np.random.default_rng(114514)
    source = _make_smooth_image(rng)
    negative = _make_smooth_image(rng)
    noise = rng.integers(-8, 9, source.shape)
    positive = np.clip(source.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return {"source": source, "positive": positive, "negative": negative}


def load_images_from_paths(paths: dict[str, Path]) -> dict[str, NumericArray]:
    """Load images from paths. Raises if any path is invalid."""
    images: dict[str, NumericArray] = {}
    for name, path in paths.items():
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Could not read {path}")
        images[name] = img
    return images
