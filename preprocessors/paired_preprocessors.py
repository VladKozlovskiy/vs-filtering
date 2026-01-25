import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Optional, Union

import numpy as np
from skimage.exposure import match_histograms

from utils.input_validation import (
    assert_ndarray,
    assert_ndim,
    assert_same_shape,
)

from .cell_preprocessors import CellImagePreprocessor

logger = logging.getLogger(__name__)


class PairedCellImagePreprocessor(ABC):
    @abstractmethod
    def __call__(
        self,
        src: np.ndarray,
        tgt: np.ndarray,
        mask_src: Optional[np.ndarray] = None,
        mask_tgt: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        pass


class PairedFromSingle(PairedCellImagePreprocessor):
    def __init__(self, preprocessor: CellImagePreprocessor):
        assert isinstance(
            preprocessor, CellImagePreprocessor
        ), f"preprocessor must be CellImagePreprocessor, got {type(preprocessor).__name__}"
        self.preprocessor = preprocessor

    def __call__(
        self,
        src: np.ndarray,
        tgt: np.ndarray,
        mask_src: Optional[np.ndarray] = None,
        mask_tgt: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert_ndarray(src, "src")
        assert_ndarray(tgt, "tgt")

        return self.preprocessor(src), self.preprocessor(tgt)


class LambdaPairedProcessor(PairedCellImagePreprocessor):
    def __init__(self, fn: Callable):
        assert callable(fn), f"fn must be callable, got {type(fn).__name__}"
        self.fn = fn

    def __call__(
        self,
        src: np.ndarray,
        tgt: np.ndarray,
        mask_src: Optional[np.ndarray] = None,
        mask_tgt: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert_ndarray(src, "src")
        assert_ndarray(tgt, "tgt")

        result = self.fn(src, tgt, mask_src, mask_tgt)

        assert (
            isinstance(result, tuple) and len(result) == 2
        ), f"Lambda function must return tuple of (src, tgt), got {type(result)}"
        return result


class MatchHistograms(PairedCellImagePreprocessor):
    def __init__(self, match_to: str = "src"):
        assert match_to in (
            "src",
            "tgt",
        ), f"match_to must be 'src' or 'tgt', got '{match_to}'"
        self.match_to = match_to

    def __call__(
        self,
        src: np.ndarray,
        tgt: np.ndarray,
        mask_src: Optional[np.ndarray] = None,
        mask_tgt: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert_ndarray(src, "src")
        assert_ndarray(tgt, "tgt")
        assert_ndim(src, (2, 3), "src")
        assert_ndim(tgt, (2, 3), "tgt")
        assert_same_shape(src, tgt, "src", "tgt")

        if src.ndim == 3:
            assert (
                src.shape[-1] == tgt.shape[-1]
            ), f"src and tgt must have same channel count, got {src.shape[-1]} and {tgt.shape[-1]}"

        channel_axis = -1 if src.ndim == 3 else None

        if self.match_to == "src":
            matched = match_histograms(tgt, src, channel_axis=channel_axis)
            return src, matched

        matched = match_histograms(src, tgt, channel_axis=channel_axis)
        return matched, tgt


class PairedCellPreprocessingPipeline(PairedCellImagePreprocessor):
    # Принимает как PairedCellImagePreprocessor, так и CellImagePreprocessor.
    # Одиночные препроцессоры автоматически оборачиваются в PairedFromSingle.

    def __init__(
        self,
        preprocessors: list[Union[PairedCellImagePreprocessor, CellImagePreprocessor]],
    ):
        assert len(preprocessors) > 0, "Pipeline must have at least one preprocessor"

        self.preprocessors: list[PairedCellImagePreprocessor] = []
        wrapped_count = 0

        for i, p in enumerate(preprocessors):
            if isinstance(p, PairedCellImagePreprocessor):
                self.preprocessors.append(p)
            elif isinstance(p, CellImagePreprocessor):
                self.preprocessors.append(PairedFromSingle(p))
                wrapped_count += 1
            else:
                raise TypeError(
                    f"preprocessors[{i}] must be CellImagePreprocessor or "
                    f"PairedCellImagePreprocessor, got {type(p).__name__}"
                )

        if wrapped_count > 0:
            logger.debug(f"Auto-wrapped {wrapped_count} single preprocessors to paired")

    def __call__(
        self,
        src: np.ndarray,
        tgt: np.ndarray,
        mask_src: Optional[np.ndarray] = None,
        mask_tgt: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert_ndarray(src, "src")
        assert_ndarray(tgt, "tgt")

        for p in self.preprocessors:
            src, tgt = p(src, tgt, mask_src, mask_tgt)

        return src, tgt
