"""Shared NumPy array type aliases for image_hash."""

import numpy as np
from numpy.typing import NDArray

type NumericArray = NDArray[np.integer | np.floating]
type FloatArray = NDArray[np.floating]
type IntegerArray = NDArray[np.integer]
type UInt8Array = NDArray[np.uint8]
