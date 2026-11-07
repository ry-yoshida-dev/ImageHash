"""
Tests for image_hash: hash computation and cross distance matrix per method.

- Pytest: uses fixtures from conftest (dummy_images, similar_images, real_images) and _helpers.
- Script: run `python -m tests.test` from the repository root for ad-hoc checks
  (e.g. --data-dir, --methods, --list-methods).
"""

import argparse
import time
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from image_hash import BitwiseHasher, VectorHasher, WaveletHash
from image_hash.hasher import ImageHasher
from image_hash.method import HashMethod
from image_hash.types import NumericArray

from ._helpers import (
    build_hasher,
    load_images_from_paths,
    make_dummy_images,
    run_cross_distance,
)

# --- Pytest ---


@pytest.mark.parametrize("method", list(HashMethod))
def test_cross_distance_dummy(
    method: HashMethod, dummy_images: dict[str, NumericArray]
) -> None:
    """Each method: build hasher, compute cross distance on dummy images; diagonal is 0."""
    hasher = build_hasher(method)
    cross_dist = run_cross_distance(hasher, dummy_images, method)
    n = len(dummy_images)
    assert cross_dist.shape == (n, n)
    np.testing.assert_array_almost_equal(np.diag(cross_dist), 0)
    np.testing.assert_array_almost_equal(cross_dist, cross_dist.T)


@pytest.mark.parametrize("method", list(HashMethod))
def test_near_duplicate_is_closer_than_unrelated(
    method: HashMethod, similar_images: dict[str, NumericArray]
) -> None:
    """Each method ranks a noisy copy of the source closer than an unrelated image."""
    hasher = build_hasher(method)
    cross_dist = run_cross_distance(hasher, similar_images, method)
    # Row 0 holds the distances from "source" to (source, positive, negative).
    assert cross_dist[0, 1] < cross_dist[0, 2]


@pytest.mark.parametrize("method", list(HashMethod))
def test_build_selects_hasher_by_bit_convertibility(method: HashMethod) -> None:
    hasher = build_hasher(method)
    expected = BitwiseHasher if method.is_bit_convertible else VectorHasher
    assert type(hasher) is expected
    assert hasher.method is method


def test_build_wavelet_without_object_warns() -> None:
    with pytest.warns(UserWarning, match="Wavelet object is not provided"):
        hasher = ImageHasher.build(HashMethod.WAVELET)
    assert isinstance(hasher.hash_obj, WaveletHash)


def test_build_non_wavelet_with_wavelet_object_warns() -> None:
    with pytest.warns(UserWarning, match="not used for non-WAVELET"):
        hasher = ImageHasher.build(HashMethod.P, wavelet_obj=WaveletHash())
    assert not isinstance(hasher.hash_obj, WaveletHash)


def test_hasher_rejects_mismatched_method() -> None:
    with pytest.raises(ValueError, match="non-bit-convertible"):
        _ = VectorHasher(method=HashMethod.P, hash_obj=HashMethod.P.object)
    with pytest.raises(ValueError, match="bit-convertible"):
        _ = BitwiseHasher(
            method=HashMethod.COLOR_MOMENT, hash_obj=HashMethod.COLOR_MOMENT.object
        )


def test_to_binary_unpacks_single_and_stacked_hashes() -> None:
    single = np.array([0b10000000, 0b00000001], dtype=np.uint8)
    np.testing.assert_array_equal(
        ImageHasher.to_binary(single),
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    )
    stacked = np.stack([single, single])
    assert ImageHasher.to_binary(stacked).shape == (2, 16)
    with pytest.raises(ValueError, match="1D"):
        _ = ImageHasher.to_binary(np.zeros((1, 1, 1), dtype=np.uint8))


@pytest.mark.parametrize(
    "method",
    [m for m in HashMethod if m is not HashMethod.WAVELET],
)
def test_hash_size_matches_computed_bytes(
    method: HashMethod, dummy_images: dict[str, NumericArray]
) -> None:
    hasher = build_hasher(method)
    assert hasher.compute(dummy_images["source"]).shape == (method.hash_size,)


def test_wavelet_hash_size_is_not_fixed() -> None:
    with pytest.raises(ValueError, match="fixed hash_size"):
        _ = HashMethod.WAVELET.hash_size


@pytest.mark.parametrize("hash_size", [8, 16])
def test_wavelet_hash_has_hash_size_squared_bits(
    hash_size: int, dummy_images: dict[str, NumericArray]
) -> None:
    wavelet_hash = WaveletHash(hash_size=hash_size)
    hash_value = wavelet_hash.compute(dummy_images["source"])
    assert hash_value.shape == (hash_size * hash_size // 8,)
    assert wavelet_hash.compare(hash_value, hash_value) == 0.0


@pytest.mark.parametrize(("hash_size", "image_scale"), [(6, None), (0, None), (8, 12)])
def test_wavelet_hash_rejects_non_power_of_two(
    hash_size: int, image_scale: int | None
) -> None:
    with pytest.raises(ValueError, match="power of 2"):
        _ = WaveletHash(hash_size=hash_size, image_scale=image_scale)


def test_wavelet_hash_rejects_image_scale_smaller_than_hash_size(
    dummy_images: dict[str, NumericArray],
) -> None:
    wavelet_hash = WaveletHash(hash_size=16, image_scale=8)
    with pytest.raises(ValueError, match="cannot be larger than image_scale"):
        _ = wavelet_hash.compute(dummy_images["source"])


def test_cross_distance_real_paths(real_images: dict[str, NumericArray]) -> None:
    """
    Run all methods on real images from data/; print timing and matrix.

    Skipped when data/source.jpg, data/positive.jpg, data/negative.jpg are not present.
    """
    for method in HashMethod:
        hasher = build_hasher(method)
        t0 = time.perf_counter()
        cross_dist = run_cross_distance(hasher, real_images, method)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert cross_dist.shape == (3, 3)
        print(f"--- method: {method.value} ---")
        print(f"elapsed time: {elapsed_ms:.2f} ms")
        print(f"cross_dist_matrix:\n{cross_dist}")


# --- Script mode: flexible CLI ---


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run image_hash cross-distance on images (dummy or from paths)."
    )
    _ = parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing source.jpg, positive.jpg, negative.jpg (default: random dummy images)",
    )
    _ = parser.add_argument(
        "--methods",
        type=str,
        default="all",
        help="Comma-separated method names or 'all' (default: all)",
    )
    _ = parser.add_argument(
        "--list-methods",
        action="store_true",
        help="Print available methods and exit",
    )
    _ = parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only print errors",
    )
    return parser.parse_args()


def _resolve_methods(names: str) -> list[HashMethod]:
    if names.strip().lower() == "all":
        return list(HashMethod)
    chosen: list[HashMethod] = []
    for raw_name in names.split(","):
        name = raw_name.strip()
        if name not in HashMethod.__members__:
            raise SystemExit(
                f"Unknown method: {name}. Use --list-methods to see options."
            )
        chosen.append(HashMethod[name])
    return chosen


def main() -> None:
    args = _parse_args()
    is_list_methods = cast(bool, args.list_methods)
    data_dir = cast(Path | None, args.data_dir)
    method_names = cast(str, args.methods)
    is_quiet = cast(bool, args.quiet)

    if is_list_methods:
        print("Available methods:", ", ".join(m.name for m in HashMethod))
        return

    if data_dir is not None:
        paths = {
            "source": data_dir / "source.jpg",
            "positive": data_dir / "positive.jpg",
            "negative": data_dir / "negative.jpg",
        }
        if not all(p.exists() for p in paths.values()):
            raise SystemExit(
                f"Missing images in {data_dir}: need source.jpg, positive.jpg, negative.jpg"
            )
        images = load_images_from_paths(paths)
        source_label = str(data_dir)
    else:
        images = make_dummy_images()
        source_label = "dummy"

    for method in _resolve_methods(method_names):
        hasher = build_hasher(method)
        t0 = time.perf_counter()
        cross_dist = run_cross_distance(hasher, images, method)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if not is_quiet:
            print(f"--- method: {method.value} (images: {source_label}) ---")
            print(f"elapsed time: {elapsed_ms:.2f} ms")
            print(f"cross_dist_matrix:\n{cross_dist}")


if __name__ == "__main__":
    main()
