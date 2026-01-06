from abc import ABC, abstractmethod
from typing import Iterable, List, Any, Callable, Optional, Tuple


class BaseQualityEvaluator(ABC):
    @abstractmethod
    def __call__(self, similarity_scores, labels):
        pass