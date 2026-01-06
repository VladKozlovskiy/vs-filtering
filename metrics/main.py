from abc import ABC, abstractmethod
from typing import Iterable, List, Any, Callable, Optional, Tuple


class BaseSimilarityEstimator(ABC):
    @abstractmethod
    def compute_scores(self, dataset) -> Any:
        pass



