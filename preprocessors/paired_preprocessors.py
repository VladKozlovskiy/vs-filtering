from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, List, Optional, Tuple

import numpy as np

from .cell_preprocessors import CellImagePreprocessor


class PairedCellImagePreprocessor(ABC):
    @abstractmethod
    def __call__(
        self,
        src: np.ndarray,
        tgt: np.ndarray,
        mask_src: Optional[np.ndarray] = None,
        mask_tgt: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass


class PairedFromSingle(PairedCellImagePreprocessor):
    def __init__(self, preprocessor: CellImagePreprocessor):
        self.preprocessor = preprocessor

    def __call__(
        self,
        src: np.ndarray,
        tgt: np.ndarray,
        mask_src: Optional[np.ndarray] = None,
        mask_tgt: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        src_out = self.preprocessor(src)
        tgt_out = self.preprocessor(tgt)
        return src_out, tgt_out


class LambdaPairedProcessor(PairedCellImagePreprocessor):
    def __init__(self, fn: Callable):
        self.fn = fn

    def __call__(self, src, tgt, mask_src=None, mask_tgt=None):
        return self.fn(src, tgt, mask_src, mask_tgt)


class PairedCellPreprocessingPipeline(PairedCellImagePreprocessor):
    def __init__(self, preprocessors: list[PairedCellImagePreprocessor]):
        self.preprocessors = preprocessors

    def __call__(self, src, tgt, mask_src=None, mask_tgt=None):
        for p in self.preprocessors:
            src, tgt = p(src, tgt, mask_src, mask_tgt)
        return src, tgt
