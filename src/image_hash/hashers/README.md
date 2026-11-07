# hashers

## Overview

Hasher implementations used by `ImageHasher.build()`.

## Components

| Component | Description |
|------------|------|
| [bitwise.py](bitwise.py) | `BitwiseHasher`: bit-convertible methods, including `WAVELET`, with Hamming distance |
| [vector.py](vector.py) | `VectorHasher`: non-bit-convertible methods (`COLOR_MOMENT`) with Euclidean distance |
