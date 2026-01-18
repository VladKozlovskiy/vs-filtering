import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from abc import ABC, abstractmethod
from typing import Iterable, List, Any, Callable, Optional, Tuple
import torch

class CellImagePreprocessor(ABC):
    @abstractmethod
    def __call__(self, image: Any) -> Any:
        pass


class LambdaPreprocessor(CellImagePreprocessor):
    def __init__(self, fn: Callable[[Any], Any]):
        self.fn = fn

    def __call__(self, image: Any) -> Any:
        return self.fn(image)

class TresholdBinarizationProcessor(CellImagePreprocessor):
    def __init__(self, threshold=None, percentile=80, buffer=5):
        self.threshold = threshold
        self.percentile = percentile
        self.buffer = buffer

    def __call__(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        threshold = self.threshold
        if threshold is None:
            threshold = np.percentile(image, self.percentile) - self.buffer

        mask = (image < threshold)
        return mask.astype(np.uint8)


class SmoothingPreprocessor(CellImagePreprocessor):
    def __init__(self, method: str = "gaussian", kernel_size: int = 5):
        assert method in ("gaussian", "median"), (
            "method must be 'gaussian' or 'median'"
        )
        assert kernel_size % 2 == 1 and kernel_size > 1, (
            "kernel_size must be odd and > 1"
        )


        self.method = method
        self.kernel_size = kernel_size

    def __call__(self, image: np.ndarray) -> np.ndarray:
        assert image.ndim in (2, 3), f"Unsupported image shape: {image.shape}"
        
        has_channels = (image.ndim == 3)
        
        match self.method:
            case "gaussian":
                sigma = 0.3 * ((self.kernel_size - 1) * 0.5 - 1) + 0.8
        
                radius = (self.kernel_size - 1) // 2
                truncate_val = radius / sigma
        
                if has_channels:
                    sigma = (sigma, sigma, 0)
        
                return gaussian_filter(
                    image,
                    sigma=sigma,
                    truncate=truncate_val,
                    mode="nearest",
                )
        
            case "median":
                size = self.kernel_size
                if has_channels:
                    size = (size, size, 1)
        
                return median_filter(
                    image,
                    size=size,
                    mode="nearest",
                )
        
            case _:
                raise ValueError("Invalid method. Choose 'gaussian' or 'median'.")

class ClahePreprocessor(CellImagePreprocessor):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self._clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size,
        )

    @staticmethod
    def _assert_not_normalized(img: np.ndarray):
        if img.dtype != np.uint8:
            assert img.max() > 1.0, (
                "CLAHE expects image in [0, 255], but got normalized image in [0, 1]. "
                "Apply normalization AFTER CLAHE."
            )

    def __call__(self, img: np.ndarray) -> np.ndarray:
        assert img.ndim in (2, 3), f"Unsupported image shape: {img.shape}"

        self._assert_not_normalized(img)

        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        if img.ndim == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            lab[..., 0] = self._clahe.apply(lab[..., 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return self._clahe.apply(img)

class CellPreprocessingPipeline:
    def __init__(self, preprocessors: Iterable[CellImagePreprocessor]):
        self.preprocessors: List[CellImagePreprocessor] = list(preprocessors)

    def __call__(self, image: Any) -> Any:
        for preprocessor in self.preprocessors:
            image = preprocessor(image)
        return image

class ToTensor(CellImagePreprocessor):
    def __call__(self, image: Any) -> torch.Tensor:
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # Добавляем канал если изображение 2D
        if image.ndim == 2:
            image = image[np.newaxis, ...]  # (1, H, W)
        elif image.ndim == 3:
            image = image.transpose((2, 0, 1))  # (C, H, W)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}. Expected 2D (H, W) or 3D (H, W, C)")
        
        image = image.astype(np.float32)
        return torch.from_numpy(image)