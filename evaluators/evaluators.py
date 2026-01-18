"""
Модуль для оценки качества метрик схожести.
"""
from abc import ABC, abstractmethod
from typing import Iterable, List, Any, Callable, Optional, Tuple, Dict

import itertools
import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
import logging

logger = logging.getLogger(__name__)


class BaseQualityEvaluator(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute_metric(self, similarity_scores: List[float], labels: List[Any]) -> Any:
        pass

    def __call__(self, similarity_scores: Dict[str, List[float]], labels: List[Any]) -> Dict[str, Any]:
        if len(labels) == 0:
            raise ValueError("Labels cannot be empty")
        
        quality_dct = {}
        for metr_name, scores in similarity_scores.items():
            if len(scores) != len(labels):
                raise ValueError(
                    f"Length mismatch: scores ({len(scores)}) != labels ({len(labels)}) for metric {metr_name}"
                )
            try:
                quality_dct[metr_name] = self.compute_metric(scores, labels)
            except Exception as e:
                logger.error(f"Error computing metric for {metr_name}: {e}")
                raise
        return quality_dct


class LambdaQualityEvaluator(BaseQualityEvaluator, ABC):
    def __init__(self, fn, name):
        super().__init__(name)
        self.fn = fn

    def compute_metric(self, similarity_scores, labels):
        return self.fn(similarity_scores, labels)
     
        

class AUCEvaluator(BaseQualityEvaluator):

    def __init__(self):
        super().__init__('MulticlassAUC')
        
    def compute_metric(self, similarity_scores: List[float], labels: List[Any]) -> Dict[str, float]:
        y_true = np.array(labels)
        y_score = np.array(similarity_scores, dtype=float)
        
        if len(y_true) == 0:
            raise ValueError("Labels cannot be empty")
        
        # Проверяем корреляцию и инвертируем если нужно
        try:
            corr, _ = spearmanr(y_true, y_score)
            if np.isnan(corr):
                logger.warning("Spearman correlation is NaN, using original scores")
            elif corr < 0:
                y_score = -y_score
                logger.info("Negative correlation detected, inverted scores")
        except Exception as e:
            logger.warning(f"Error computing correlation: {e}, using original scores")
    
        unique_classes = np.sort(np.unique(y_true))
        n_classes = len(unique_classes)
        
        if n_classes < 2:
            raise ValueError(f"Need at least 2 classes, got {n_classes}")
    
        # Бинарная классификация
        if n_classes == 2:
            try:
                auc = roc_auc_score(y_true, y_score)
                return {self.name: float(auc)}
            except Exception as e:
                logger.error(f"Error computing binary AUC: {e}")
                raise
    
        # Мультиклассовая классификация: вычисляем AUC для каждой пары классов
        pair_aucs = []
        for c1, c2 in itertools.combinations(unique_classes, 2):
            mask = np.isin(y_true, [c1, c2])
            
            if np.sum(mask) < 2:
                logger.warning(f"Not enough samples for pair ({c1}, {c2}), skipping")
                continue
            
            y_true_subset = y_true[mask]
            y_score_subset = y_score[mask]
    
            y_true_binary = (y_true_subset == c2).astype(int)
    
            try:
                pair_auc = roc_auc_score(y_true_binary, y_score_subset)
                if not np.isnan(pair_auc):
                    pair_aucs.append(pair_auc)
            except Exception as e:
                logger.warning(f"Error computing AUC for pair ({c1}, {c2}): {e}")
                continue

        if len(pair_aucs) == 0:
            raise ValueError("Could not compute any valid pair AUCs")

        return {self.name: float(np.mean(pair_aucs))}     

class SequencialScoreEvaluator:
    def __init__(self, evaluators):
        self.evaluators = evaluators

    def __call__(self, similarity_scores, labels) -> Any:
        evaluation_res = {}
        for evaluator in self.evaluators:
            evaluation_res.update(evaluator(similarity_scores, labels))
        return evaluation_res