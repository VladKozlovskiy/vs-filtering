from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Any

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, median_filter
from skimage.color import rgb2hed

from utils.input_validation import (
    assert_dtype,
    assert_grayscale,
    assert_ndarray,
    assert_ndim,
    assert_range,
    assert_rgb,
)


class CellImagePreprocessor(ABC):
    @abstractmethod
    def __call__(self, image: np.ndarray) -> Any:
        pass


class CellPreprocessingPipeline:
    def __init__(self, preprocessors: Iterable[CellImagePreprocessor]):
        self.preprocessors: list[CellImagePreprocessor] = list(preprocessors)
        assert (
            len(self.preprocessors) > 0
        ), "Pipeline must have at least one preprocessor"

    def __call__(self, image: np.ndarray) -> Any:
        for preprocessor in self.preprocessors:
            image = preprocessor(image)
        return image


class LambdaPreprocessor(CellImagePreprocessor):
    def __init__(self, fn: Callable[[Any], Any]):
        assert callable(fn), f"fn must be callable, got {type(fn).__name__}"
        self.fn = fn

    def __call__(self, image: np.ndarray) -> Any:
        return self.fn(image)


class ToGrayScale(CellImagePreprocessor):
    def __init__(self, channel: str = "rgb"):
        assert (
            channel == "rgb"
        ), f"ToGrayScale supports only 'rgb' channel, got '{channel}'"
        self.channel = channel

    def __call__(self, img: np.ndarray) -> np.ndarray:
        assert_ndarray(img, "img")

        if img.ndim == 2:
            return img

        assert_rgb(img, "img")
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


class ToUint8(CellImagePreprocessor):
    # Input: float32/float64 in [0, 1] OR uint8 in [0, 255]
    # Output: uint8 in [0, 255]

    def __call__(self, img: np.ndarray) -> np.ndarray:
        assert_ndarray(img, "img")
        assert_ndim(img, (2, 3), "img")

        if img.dtype == np.uint8:
            return img

        if img.dtype in (np.float32, np.float64):
            assert_range(img, 0.0, 1.0, "img (float)")
            return (img * 255.0).round().astype(np.uint8)

        raise ValueError(f"ToUint8 expects uint8 or float32/float64, got {img.dtype}")


class ToTensor(CellImagePreprocessor):
    # Input: np.ndarray (H, W) or (H, W, C)
    # Output: torch.Tensor (C, H, W) float32

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        assert_ndarray(image, "image")
        assert_ndim(image, (2, 3), "image")

        if image.ndim == 2:
            image = image[np.newaxis, ...]
        else:
            assert image.shape[-1] in (
                1,
                3,
            ), f"Expected 1 or 3 channels, got {image.shape[-1]}"
            image = image.transpose((2, 0, 1))

        return torch.from_numpy(image.astype(np.float32))


class MinMaxNormalizer(CellImagePreprocessor):
    # Input: any dtype, any range
    # Output: float32 in [0, 1]

    def __init__(self, eps: float = 1e-20):
        assert eps > 0, f"eps must be positive, got {eps}"
        self.eps = eps

    def __call__(self, image: np.ndarray) -> np.ndarray:
        assert_ndarray(image, "image")
        assert_ndim(image, (2, 3), "image")

        img = image.astype(np.float32)
        img_min = img.min()
        img_max = img.max()
        return (img - img_min) / (img_max - img_min + self.eps)


class ScaleToMinusOneOne(CellImagePreprocessor):
    # Input: float in [0, 1]
    # Output: float32 in [-1, 1]

    def __call__(self, image: np.ndarray) -> np.ndarray:
        assert_ndarray(image, "image")
        assert_ndim(image, (2, 3), "image")
        assert_range(image, 0.0, 1.0, "image")

        return image.astype(np.float32) * 2.0 - 1.0


class FlipIntensity(CellImagePreprocessor):
    # Input: float in [0, 1]
    # Output: float32 in [0, 1]

    def __call__(self, image: np.ndarray) -> np.ndarray:
        assert_ndarray(image, "image")
        assert_ndim(image, (2, 3), "image")
        assert_range(image, 0.0, 1.0, "image")

        return 1.0 - image.astype(np.float32)


class ClahePreprocessor(CellImagePreprocessor):
    # Input: uint8 in [0, 255]
    # Output: uint8 in [0, 255]

    def __init__(
        self, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)
    ):
        assert clip_limit > 0, f"clip_limit must be positive, got {clip_limit}"
        assert (
            len(tile_grid_size) == 2
        ), f"tile_grid_size must be (h, w), got {tile_grid_size}"
        assert all(
            s > 0 for s in tile_grid_size
        ), "tile_grid_size values must be positive"

        self._clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size,
        )

    def __call__(self, img: np.ndarray) -> np.ndarray:
        assert_ndarray(img, "img")
        assert_ndim(img, (2, 3), "img")
        assert_dtype(img, (np.uint8,), "img")

        if img.ndim == 3:
            assert_rgb(img, "img")
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            lab[..., 0] = self._clahe.apply(lab[..., 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return self._clahe.apply(img)


class SmoothingPreprocessor(CellImagePreprocessor):
    def __init__(self, method: str = "gaussian", kernel_size: int = 5):
        assert method in (
            "gaussian",
            "median",
        ), f"method must be 'gaussian' or 'median', got '{method}'"
        assert (
            kernel_size % 2 == 1 and kernel_size > 1
        ), f"kernel_size must be odd and > 1, got {kernel_size}"

        self.method = method
        self.kernel_size = kernel_size

    def __call__(self, image: np.ndarray) -> np.ndarray:
        assert_ndarray(image, "image")
        assert_ndim(image, (2, 3), "image")

        has_channels = image.ndim == 3

        if self.method == "gaussian":
            sigma = 0.3 * ((self.kernel_size - 1) * 0.5 - 1) + 0.8
            radius = (self.kernel_size - 1) // 2
            truncate_val = radius / sigma

            if has_channels:
                sigma = (sigma, sigma, 0)

            return gaussian_filter(
                image, sigma=sigma, truncate=truncate_val, mode="nearest"
            )

        size = self.kernel_size
        if has_channels:
            size = (size, size, 1)

        return median_filter(image, size=size, mode="nearest")


class TresholdBinarizationProcessor(CellImagePreprocessor):
    def __init__(
        self, threshold: float = None, percentile: float = 80, buffer: float = 5
    ):
        assert (
            percentile >= 0 and percentile <= 100
        ), f"percentile must be in [0, 100], got {percentile}"
        self.threshold = threshold
        self.percentile = percentile
        self.buffer = buffer

    def __call__(self, image: np.ndarray) -> np.ndarray:
        assert_ndarray(image, "image")
        assert_grayscale(image, "image")

        threshold = self.threshold
        if threshold is None:
            threshold = np.percentile(image, self.percentile) - self.buffer

        return (image < threshold).astype(np.uint8)


class ToRGB(CellImagePreprocessor):
    # Input: (H, W) or (H, W, 1) or (H, W, 3)
    # Output: (H, W, 3) — grayscale дублируется в 3 канала

    def __call__(self, image: np.ndarray) -> np.ndarray:
        assert_ndarray(image, "image")
        assert_ndim(image, (2, 3), "image")

        if image.ndim == 2:
            return np.stack([image, image, image], axis=-1)

        if image.shape[-1] == 1:
            return np.concatenate([image, image, image], axis=-1)

        if image.shape[-1] == 3:
            return image

        raise ValueError(f"ToRGB expects 1 or 3 channels, got {image.shape[-1]}")


class HEDChannelExtractor(CellImagePreprocessor):
    # Input: RGB float in [0, 1]
    # Output: grayscale float32

    VALID_MODES = ("hematoxylin", "eosin", "dab")

    def __init__(self, mode: str = "hematoxylin"):
        assert (
            mode in self.VALID_MODES
        ), f"mode must be one of {self.VALID_MODES}, got '{mode}'"
        self.mode = mode

    def __call__(self, img_rgb: np.ndarray) -> np.ndarray:
        assert_ndarray(img_rgb, "img_rgb")
        assert_rgb(img_rgb, "img_rgb")
        assert_range(img_rgb, 0.0, 1.0, "img_rgb")

        hed = rgb2hed(img_rgb.astype(np.float32))

        match self.mode:
            case "hematoxylin":
                ch = hed[..., 0]
            case "eosin":
                ch = hed[..., 1]
            case _:
                ch = hed[..., 2]

        return (1 - ch).astype(np.float32)
