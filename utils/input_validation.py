"""Функции валидации входных данных."""


import numpy as np
import torch


def assert_ndarray(image, name: str = "image") -> None:
    assert isinstance(
        image, np.ndarray
    ), f"{name} must be np.ndarray, got {type(image).__name__}"


def assert_ndim(
    image: np.ndarray, allowed: tuple[int, ...], name: str = "image"
) -> None:
    assert (
        image.ndim in allowed
    ), f"{name} must have ndim in {allowed}, got ndim={image.ndim}, shape={image.shape}"


def assert_rgb(image: np.ndarray, name: str = "image") -> None:
    assert (
        image.ndim == 3 and image.shape[-1] == 3
    ), f"{name} must be RGB with shape (H, W, 3), got {image.shape}"


def assert_grayscale(image: np.ndarray, name: str = "image") -> None:
    assert (
        image.ndim == 2
    ), f"{name} must be grayscale with shape (H, W), got {image.shape}"


def assert_range(
    image: np.ndarray,
    low: float,
    high: float,
    name: str = "image",
    tolerance: float = 0.01,
) -> None:
    img_min, img_max = float(image.min()), float(image.max())
    assert (
        img_min >= low - tolerance and img_max <= high + tolerance
    ), f"{name} values must be in [{low}, {high}], got [{img_min:.4f}, {img_max:.4f}]"


def assert_dtype(image: np.ndarray, dtypes: tuple, name: str = "image") -> None:
    assert (
        image.dtype in dtypes
    ), f"{name} must have dtype in {dtypes}, got {image.dtype}"


def assert_same_shape(
    arr1: np.ndarray, arr2: np.ndarray, name1: str, name2: str
) -> None:
    assert (
        arr1.shape == arr2.shape
    ), f"{name1} and {name2} must have same shape, got {arr1.shape} and {arr2.shape}"


def assert_same_ndim(
    arr1: np.ndarray, arr2: np.ndarray, name1: str, name2: str
) -> None:
    assert (
        arr1.ndim == arr2.ndim
    ), f"{name1} and {name2} must have same ndim, got {arr1.ndim} and {arr2.ndim}"


def assert_tensor(x, name: str = "tensor") -> None:
    assert isinstance(
        x, torch.Tensor
    ), f"{name} must be torch.Tensor, got {type(x).__name__}"


def assert_tensor_shape(x: torch.Tensor, ndim: int, name: str = "tensor") -> None:
    assert x.ndim == ndim, f"{name} must be {ndim}D, got {x.ndim}D with shape {x.shape}"
