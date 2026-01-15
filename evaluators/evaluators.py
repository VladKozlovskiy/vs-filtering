from abc import ABC, abstractmethod
from typing import Iterable, List, Any, Callable, Optional, Tuple

import itertools
import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

class BaseQualityEvaluator(ABC):
    @abstractmethod
    def __call__(self, similarity_scores, labels):
        pass


class LambdaQualityEvaluator(BaseSimilarityEstimator):
    def __init__(self, fn: Callable[[Any], Any], name):
        self.fn = fn
        self.name = name
    def __call__(self, similarity_scores, labels) -> Any:
        return {self.name : self.fn(similarity_scores, labels)} 

class AUCEvaluator(BaseSimilarityEstimator):
    def __init__(self):
        self.name = 'MulticlassAUC'
        
    def __call__(self, similarity_scores, labels) -> Any:
        y_true = np.array(labels)
        y_score = np.array(similarity_scores, dtype=float)

        corr, _ = spearmanr(y_true, y_score)
        if corr < 0:
            # Через логгер переделать 
            #if verbose:
            #    print(f"[INFO] Обнаружена отрицательная корреляция ({corr:.3f}). Инвертируем метрику.")
            y_score = -y_score
    
        unique_classes = np.sort(np.unique(y_true))
        n_classes = len(unique_classes)
    
        if n_classes == 2:
            return roc_auc_score(y_true, y_score)
    
        pair_aucs = []
        for c1, c2 in itertools.combinations(unique_classes, 2):
            mask = np.isin(y_true, [c1, c2])
    
            y_true_subset = y_true[mask]
            y_score_subset = y_score[mask]
    
            y_true_binary = (y_true_subset == c2).astype(int)
    
            pair_auc = roc_auc_score(y_true_binary, y_score_subset)
            pair_aucs.append(pair_auc)
            # Через логгер переделать 
            #if verbose:
            #    print(f"   AUC для пары {c1} vs {c2}: {pair_auc:.4f}")
    
        return {self.name : np.mean(pair_aucs)}     

class SequencialScoreEvaluator:
    def __init__(self, evaluators):
        self.evaluators = evaluators

    def __call__(self, similarity_scores, labels) -> Any:
        evaluation_res = {}
        for evaluator in self.evaluators:
            evaluation_res.update(evaluator(similarity_scores, labels))
        return evaluation_res