from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, median_filter
from skimage.color import rgb2hed


class CellImagePreprocessor(ABC):
    @abstractmethod
    def __call__(self, image: Any) -> Any:
        pass


class CellPreprocessingPipeline:
    def __init__(self, preprocessors: Iterable[CellImagePreprocessor]):
        self.preprocessors: List[CellImagePreprocessor] = list(preprocessors)

    def __call__(self, image: Any) -> Any:
        for preprocessor in self.preprocessors:
            image = preprocessor(image)
        return image


class LambdaPreprocessor(CellImagePreprocessor):
    def __init__(self, fn: Callable[[Any], Any]):
        self.fn = fn

    def __call__(self, image: Any) -> Any:
        return self.fn(image)


class ToGrayScale(CellImagePreprocessor):
    def __init__(self, channel: str = "rgb"):
        self.channel = channel

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if not isinstance(img, np.ndarray):
            img = np.array(img)

        if img.ndim == 2:
            return img

        if img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError(
                f"Expected RGB image with shape (H, W, 3), got {img.shape}"
            )

        if self.channel != "rgb":
            raise ValueError("ToGrayScale supports only RGB input")

        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


class ToUint8(CellImagePreprocessor):
    def __init__(self, assume_range: str = "0_1"):
        self.assume_range = assume_range

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if not isinstance(img, np.ndarray):
            img = np.array(img)

        if self.assume_range not in ("0_1", "0_255"):
            raise ValueError("assume_range must be '0_1' or '0_255'")

        if self.assume_range == "0_1":
            if img.min() < 0.0 or img.max() > 1.0:
                raise ValueError(
                    "ToUint8 expects image in [0, 1] when assume_range='0_1'"
                )
            img = (img * 255.0).round()
        else:
            if img.min() < 0.0 or img.max() > 255.0:
                raise ValueError(
                    "ToUint8 expects image in [0, 255] when assume_range='0_255'"
                )

        return img.astype(np.uint8)


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
            raise ValueError(
                f"Unsupported image shape: {image.shape}. Expected 2D (H, W) or 3D (H, W, C)"
            )

        image = image.astype(np.float32)
        return torch.from_numpy(image)


class TresholdBinarizationProcessor(CellImagePreprocessor):
    def __init__(self, threshold=None, percentile=80, buffer=5):
        self.threshold = threshold
        self.percentile = percentile
        self.buffer = buffer

    def __call__(self, image):
        if len(image.shape) == 3:
            raise ValueError(
                "TresholdBinarizationProcessor expects grayscale image. "
                "Use ToGrayScale before this processor."
            )

        threshold = self.threshold
        if threshold is None:
            threshold = np.percentile(image, self.percentile) - self.buffer

        mask = image < threshold
        return mask.astype(np.uint8)


class SmoothingPreprocessor(CellImagePreprocessor):
    def __init__(self, method: str = "gaussian", kernel_size: int = 5):
        assert method in ("gaussian", "median"), "method must be 'gaussian' or 'median'"
        assert (
            kernel_size % 2 == 1 and kernel_size > 1
        ), "kernel_size must be odd and > 1"

        self.method = method
        self.kernel_size = kernel_size

    def __call__(self, image: np.ndarray) -> np.ndarray:
        assert image.ndim in (2, 3), f"Unsupported image shape: {image.shape}"

        has_channels = image.ndim == 3

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
    def _assert_uint8(img: np.ndarray):
        if img.dtype != np.uint8:
            raise ValueError(
                "CLAHE expects uint8 image in [0, 255]. "
                "Use ToUint8 or normalize AFTER CLAHE."
            )

    def __call__(self, img: np.ndarray) -> np.ndarray:
        assert img.ndim in (2, 3), f"Unsupported image shape: {img.shape}"

        self._assert_uint8(img)

        if img.ndim == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            lab[..., 0] = self._clahe.apply(lab[..., 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return self._clahe.apply(img)


class MinMaxNormalizer(CellImagePreprocessor):
    """
    Min-Max нормализация для произвольных изображений.
    Нормализует значения изображения в диапазон [0, 1] используя min-max нормализацию.
    Сохраняет исходную структуру изображения (2D или 3D).
    """

    def __init__(self, eps: float = 1e-20):
        self.eps = eps

    def __call__(self, image: Any) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        img = image.astype(np.float32)
        # Min-max нормализация
        img_min = img.min()
        img_max = img.max()
        img = img - img_min
        img = img / (img_max - img_min + self.eps)

        return img


class HEDChannelExtractor(CellImagePreprocessor):
    """
    Извлекает выбранный канал из HED пространства без нормировки.
    Ожидает RGB в диапазоне [0, 1] (float).
    """

    def __init__(self, mode: str = "hematoxylin"):
        self.mode = mode

    @staticmethod
    def _assert_range(img: np.ndarray) -> None:
        img_min = float(np.min(img))
        img_max = float(np.max(img))
        if img_min < 0.0 or img_max > 1.0:
            raise ValueError(
                "HEDChannelExtractor expects image in [0, 1]. "
                "Apply explicit normalization, e.g. MinMaxNormalizer or "
                "LambdaPreprocessor(lambda x: x / 255.0)."
            )

    def __call__(self, img_rgb: np.ndarray) -> np.ndarray:
        if not isinstance(img_rgb, np.ndarray):
            img_rgb = np.array(img_rgb)

        if img_rgb.ndim != 3 or img_rgb.shape[-1] != 3:
            raise ValueError(
                f"Expected RGB image with shape (H, W, 3), got {img_rgb.shape}"
            )

        img = img_rgb.astype(np.float32, copy=False)
        self._assert_range(img)

        hed = rgb2hed(img)  # H, E, D channels in optical density space
        match self.mode:
            case "hematoxylin":
                ch = hed[..., 0]
            case "eosin":
                ch = hed[..., 1]
            case "dab":
                ch = hed[..., 2]
            case _:
                raise ValueError(
                    f"mode must be 'hematoxylin'|'eosin'|'dab', got {self.mode}"
                )

        # In HED, stronger stain typically corresponds to more negative values.
        # Invert sign to get a positive-intensity map (no normalization applied).
        return (-ch).astype(np.float32)
