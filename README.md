# ImageHash

## Overview

Compute perceptual hashes of images and measure distances between them. Methods are selected with `HashMethod`.

| `HashMethod` | Algorithm | Distance |
|--------------|-----------|----------|
| `P` | PHash | Hamming |
| `AVERAGE` | AverageHash | Hamming |
| `BLOCK_MEAN` | BlockMeanHash | Hamming |
| `MARR_HILDRETH` | MarrHildrethHash | Hamming |
| `RADIAL_VARIANCE` | RadialVarianceHash | Hamming |
| `WAVELET` | WaveletHash | Hamming |
| `COLOR_MOMENT` | ColorMomentHash | Euclidean |

## Installation

Requires Python 3.12+.

```bash
pip install -r requirements.txt
```

## Usage

```python
import cv2
from image_hash import HashMethod, ImageHasher

image = cv2.imread("path/to/image.jpg")  # BGR or grayscale

hasher = ImageHasher.build(HashMethod.P)
hash_value = hasher.compute(image)  # e.g. 8-byte ndarray
hash_bits = ImageHasher.to_binary(hash_value)  # (64,) for 64-bit methods
```

## Development

[tests/test.py](tests/test.py) also runs as a script that prints cross-distance matrices and timing for `source.jpg`, `positive.jpg`, and `negative.jpg` in a data directory; see `python tests/test.py --help`.
