# image_hash

## Overview

Core logic for image perceptual hashing: method selection, hasher factory, and distance measurement.

## Components

| Component | Description |
|------------|------|
| [types.py](types.py) | Shared NumPy array type aliases (`NumericArray`, `FloatArray`, `IntegerArray`) |
| [method.py](method.py) | `HashMethod` enum and method metadata |
| [hasher.py](hasher.py) | `ImageHasher` base class and factory |
| [hashers/](hashers/README.md) | Bit-based (Hamming) and vector-based (Euclidean) hasher implementations |
| [wavelet/](wavelet/README.md) | Wavelet hash implementation (`WaveletHash`, `WaveletMode`) |
