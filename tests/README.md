# tests

## Overview

Pytest suite for `image_hash`: hash computation and cross-distance matrices for every method.

## Components

| Component | Description |
|------------|------|
| [conftest.py](conftest.py) | Dummy, synthetic near-duplicate, and real image fixtures (`source`, `positive`, `negative`) |
| [_helpers.py](_helpers.py) | Hasher construction, image generation, and cross-distance helpers shared by pytest and script mode |
| [test.py](test.py) | Tests, also runnable as `python -m tests.test` with `--data-dir` and `--methods` |
