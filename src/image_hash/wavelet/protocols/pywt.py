from __future__ import annotations

from typing import Protocol, cast

import pywt  # type: ignore[import-untyped]

from ...types import FloatArray

type DetailCoeff2D = tuple[FloatArray, FloatArray, FloatArray]
type Coeffs2D = list[FloatArray | DetailCoeff2D]


class WaveletSpec(Protocol):
    dec_len: int


class PywtModule(Protocol):
    def wavedec2(
        self,
        data: FloatArray,
        wavelet: str,
        mode: str = "symmetric",
        level: int | None = None,
        axes: tuple[int, int] = (-2, -1),
    ) -> Coeffs2D: ...

    def waverec2(
        self,
        coeffs: Coeffs2D,
        wavelet: str,
        mode: str = "symmetric",
        axes: tuple[int, int] = (-2, -1),
    ) -> FloatArray: ...

    def Wavelet(self, name: str) -> WaveletSpec: ...

    def dwt_max_level(self, data_len: int, filter_len: int) -> int: ...


PYWT = cast(PywtModule, pywt)
