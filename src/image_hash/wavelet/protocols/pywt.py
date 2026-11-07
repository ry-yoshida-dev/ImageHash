from __future__ import annotations

from typing import Protocol, cast

# PyWavelets ships neither type hints nor stubs; PywtModule below types the subset used here.
import pywt  # type: ignore[import-untyped]  # pyright: ignore[reportMissingTypeStubs]

from ...types import FloatArray

type DetailCoeff2D = tuple[FloatArray, FloatArray, FloatArray]
type Coeffs2D = list[FloatArray | DetailCoeff2D]


class WaveletSpec(Protocol):
    """Typed view of the `pywt.Wavelet` attributes used by this package."""

    dec_len: int


class PywtModule(Protocol):
    """Typed view of the `pywt` functions used by this package."""

    def wavedec2(
        self,
        data: FloatArray,
        wavelet: str,
        mode: str = "symmetric",
        level: int | None = None,
        axes: tuple[int, int] = (-2, -1),
    ) -> Coeffs2D:
        """Run a multilevel 2D discrete wavelet decomposition."""
        ...

    def waverec2(
        self,
        coeffs: Coeffs2D,
        wavelet: str,
        mode: str = "symmetric",
        axes: tuple[int, int] = (-2, -1),
    ) -> FloatArray:
        """Reconstruct a 2D signal from multilevel wavelet coefficients."""
        ...

    def Wavelet(self, name: str) -> WaveletSpec:
        """Create a wavelet object by name."""
        ...

    def dwt_max_level(self, data_len: int, filter_len: int) -> int:
        """Return the maximum useful decomposition level."""
        ...


# `pywt.Wavelet` is defined in a compiled extension that pyright cannot see, so the module
# does not match the protocol structurally even though it provides every member at runtime.
PYWT = cast(PywtModule, pywt)  # pyright: ignore[reportInvalidCast]
