"""
Модуль для оценки качества метрик схожести.
"""

import itertools
import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def pearson_corr(similarity_scores: list[float], labels: list[Any]) -> float:
    corr, _ = pearsonr(similarity_scores, labels)
    return float(corr)


def spearman_corr(similarity_scores: list[float], labels: list[Any]) -> float:
    corr, _ = spearmanr(similarity_scores, labels)
    return float(corr)


def kendall_corr(similarity_scores: list[float], labels: list[Any]) -> float:
    corr, _ = kendalltau(similarity_scores, labels)
    return float(corr)


class BaseQualityEvaluator(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute_metric(self, similarity_scores: list[float], labels: list[Any]) -> Any:
        pass

    def __call__(
        self, similarity_scores: dict[str, list[float]], labels: list[Any]
    ) -> dict[str, Any]:
        if len(labels) == 0:
            raise ValueError("Labels cannot be empty")

        quality_dct = {}
        for metr_name, scores in similarity_scores.items():
            if len(scores) != len(labels):
                raise ValueError(
                    f"Length mismatch: scores ({len(scores)}) != labels ({len(labels)}) for metric {metr_name}"
                )
            try:
                result = self.compute_metric(scores, labels)
                # Если результат - словарь, распаковываем его с префиксом названия метрики
                if isinstance(result, dict):
                    for key, value in result.items():
                        # Ключ уже содержит self.name, просто добавляем префикс метрики
                        quality_dct[f"{metr_name}_{key}"] = value
                else:
                    # Если результат не словарь, сохраняем как есть с названием метрики
                    quality_dct[f"{metr_name}_{self.name}"] = result
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
        super().__init__("MulticlassAUC")

    def compute_metric(
        self, similarity_scores: list[float], labels: list[Any]
    ) -> dict[str, float]:
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


class MeanRecallEvaluator(BaseQualityEvaluator):
    """
    Эвалюатор для вычисления оптимальных порогов и среднего recall.
    Находит оптимальные пороги для бинарной или тройной классификации,
    максимизируя средний recall по классам.

    ВАЖНО: Ограничение, которое фиксируем как допущение
    Случаи с одинаковыми значениями метрики (ties) явно не обрабатываются.
    Это может завышать mRecall, потому что оптимизация "по индексам" и стабильная
    сортировка могут искусственно дать хороший разрез даже для неинформативной метрики
    (пример: константная метрика или мало уникальных значений).
    Для строгой обработки нужен отдельный tie-break или оптимизация по уникальным значениям метрики.
    """

    def __init__(self, method="spearman", positive_thr=0.1, eps=1e-8):
        """
        Args:
            method: метод корреляции для предобработки ('pearson' или 'spearman')
            positive_thr: порог положительной корреляции для инверсии метрики
            eps: малое значение для граничных случаев при вычислении порогов
        """
        super().__init__("MeanRecall")
        self.method = method
        self.positive_thr = positive_thr
        self.eps = eps

    def _preprocess_metric_for_thresholds(self, metric_list, target_values):
        """
        Preprocess metric values to ensure compatibility with _find_threshold_ functions:
        !!High values correspond to class 0 (best class), Low values correspond to class 2 (worst class)!!
        If correlation analysis shows the opposite relationship, it inverts the metric values.

        Args:
            metric_list: список значений метрики
            target_values: список целевых классов

        Returns:
            transformed_metric: предобработанные значения метрики (инвертированные если нужно)
            inverted: флаг инверсии
            correlation: коэффициент корреляции
        """
        metric_arr = np.array(metric_list)
        target_arr = np.array(target_values)

        # Calculate correlation
        if self.method == "pearson":
            corr, _ = pearsonr(metric_arr, target_arr)
        elif self.method == "spearman":
            corr, _ = spearmanr(metric_arr, target_arr)
        else:
            raise ValueError(
                f"Unknown method: {self.method}. Use 'pearson' or 'spearman'"
            )

        # Determine if inversion is needed
        inverted = False
        transformed_metric = metric_arr.copy()

        # Inversion needed if positive correlation exists (high metric -> high target/class)
        if not np.isnan(corr) and corr > self.positive_thr:
            transformed_metric = -1 * metric_arr
            inverted = True
            logger.debug(
                f"Metric inverted due to positive correlation ({self.method} r={corr:.3f})"
            )

        return transformed_metric, inverted, corr

    def _cut_to_thr(self, m, idx):
        """
        Вычисляет порог для разреза на индексе idx.

        Args:
            m: 1D массив трансформированных значений метрики, отсортированных по убыванию (high -> class 0)
            idx: индекс разреза в [0..n], определяющий разделение между [0..idx-1] и [idx..n-1]

        Returns:
            пороговое значение
        """
        if idx == 0:
            return m[0] + self.eps  # class0 пуст (или class1 пуст для t2)
        if idx == len(m):
            return m[-1] - self.eps  # class0 = все (или class2 пуст для t2)
        return 0.5 * (m[idx - 1] + m[idx])

    def _find_threshold_binary(self, class_list, metric_list):
        """
        Поиск оптимального порога классификации по метрике для двух классов.

        Args:
            class_list: список классов
            metric_list: список значений метрики

        Returns:
            threshold: оптимальный порог
            best_mean_recall: лучший средний recall
            best_recalls: tuple (recall0, recall1)
        """
        # Preprocess metric
        transformed_metric, inverted, _ = self._preprocess_metric_for_thresholds(
            metric_list, class_list
        )

        # Создаем DataFrame и сортируем по убыванию метрики
        df = pd.DataFrame({"class": class_list, "metric": transformed_metric})
        df = df.sort_values("metric", ascending=False).reset_index(drop=True)
        n = len(df)

        total_counts = df["class"].value_counts()
        total0 = total_counts.get(0, 0)
        total1 = total_counts.get(1, 0)

        cum0, cum1 = np.zeros(n + 1), np.zeros(n + 1)

        for i in range(1, n + 1):
            row_class = df.iloc[i - 1]["class"]
            cum0[i] = cum0[i - 1] + (1 if row_class == 0 else 0)
            cum1[i] = cum1[i - 1] + (1 if row_class == 1 else 0)

        best_mean_recall = -1
        best_i = 0
        best_recalls = (0.0, 0.0)

        for i in range(n + 1):
            recall0 = cum0[i] / total0 if total0 > 0 else 0
            recall1 = (cum1[n] - cum1[i]) / total1 if total1 > 0 else 0
            mean_recall = (recall0 + recall1) / 2

            if mean_recall > best_mean_recall:
                best_mean_recall = mean_recall
                best_recalls = (recall0, recall1)
                best_i = i

        # Определение порога
        m = df["metric"].to_numpy()
        threshold = self._cut_to_thr(m, best_i)

        # Convert threshold back to original space if inverted
        threshold = -threshold if inverted else threshold

        return threshold, best_mean_recall, best_recalls

    def _find_threshold_trinary(self, class_list, metric_list):
        """
        Поиск оптимальных порогов классификации по метрике для трех классов.

        Args:
            class_list: список классов
            metric_list: список значений метрики

        Returns:
            t1, t2: оптимальные пороги
            best_mean_recall: лучший средний recall
            best_recalls: tuple (recall0, recall1, recall2)
        """
        # Preprocess metric
        transformed_metric, inverted, _ = self._preprocess_metric_for_thresholds(
            metric_list, class_list
        )

        # Создаем DataFrame и сортируем по убыванию метрики
        df = pd.DataFrame({"class": class_list, "metric": transformed_metric})
        df = df.sort_values("metric", ascending=False).reset_index(drop=True)
        n = len(df)

        # Рассчитываем общее количество объектов в каждом классе
        total_counts = df["class"].value_counts()
        total0 = total_counts.get(0, 0)
        total1 = total_counts.get(1, 0)
        total2 = total_counts.get(2, 0)

        # Инициализация кумулятивных сумм
        cum0, cum1, cum2 = np.zeros(n + 1), np.zeros(n + 1), np.zeros(n + 1)

        # Заполнение кумулятивных сумм
        for i in range(1, n + 1):
            row_class = df.iloc[i - 1]["class"]
            cum0[i] = cum0[i - 1] + (1 if row_class == 0 else 0)
            cum1[i] = cum1[i - 1] + (1 if row_class == 1 else 0)
            cum2[i] = cum2[i - 1] + (1 if row_class == 2 else 0)

        # Вычисление массива S[j] = cum1[j]/total1 - cum2[j]/total2
        S = np.zeros(n + 1)
        for j in range(n + 1):
            term1 = cum1[j] / total1 if total1 > 0 else 0
            term2 = cum2[j] / total2 if total2 > 0 else 0
            S[j] = term1 - term2

        # Поиск правых максимумов - оптимальный t2 при фикс. t1
        right_max, right_argmax = np.zeros(n + 1), np.zeros(n + 1, dtype=int)
        right_max[n], right_argmax[n] = S[n], n

        for j in range(n - 1, -1, -1):
            if S[j] > right_max[j + 1]:
                right_max[j] = S[j]
                right_argmax[j] = j
            else:
                right_max[j] = right_max[j + 1]
                right_argmax[j] = right_argmax[j + 1]

        # Поиск оптимальных порогов
        best_mean_recall = -1
        best_i, best_j = 0, 0
        best_recalls = (0.0, 0.0, 0.0)

        for i in range(n + 1):
            j = right_argmax[i]

            # Рассчет recall для каждого класса
            recall0 = cum0[i] / total0 if total0 > 0 else 0
            recall1 = (cum1[j] - cum1[i]) / total1 if total1 > 0 else 0
            recall2 = (total2 - cum2[j]) / total2 if total2 > 0 else 0
            mean_recall = (recall0 + recall1 + recall2) / 3

            if mean_recall > best_mean_recall:
                best_mean_recall = mean_recall
                best_recalls = (recall0, recall1, recall2)
                best_i, best_j = i, j

        # Определение порогов
        m = df["metric"].to_numpy()
        t1 = self._cut_to_thr(m, best_i)
        t2 = self._cut_to_thr(m, best_j)

        # Convert thresholds back to original space if inverted
        t1, t2 = (-t1, -t2) if inverted else (t1, t2)

        return t1, t2, best_mean_recall, best_recalls

    def compute_metric(
        self, similarity_scores: list[float], labels: list[Any]
    ) -> dict[str, Any]:
        """
        Вычисляет оптимальные пороги и средний recall.

        Args:
            similarity_scores: список оценок схожести
            labels: список меток (ground truth)

        Returns:
            словарь с результатами: mean_recall, thresholds, recalls
        """
        y_true = np.array(labels)
        y_score = np.array(similarity_scores, dtype=float)

        if len(y_true) == 0:
            raise ValueError("Labels cannot be empty")

        unique_classes = np.sort(np.unique(y_true))
        n_classes = len(unique_classes)

        if n_classes < 2:
            raise ValueError(f"Need at least 2 classes, got {n_classes}")

        # Бинарная классификация
        if n_classes == 2:
            threshold, mean_recall, recalls = self._find_threshold_binary(
                list(y_true), list(y_score)
            )
            return {
                f"{self.name}_mean": float(mean_recall),
                f"{self.name}_threshold": float(threshold),
                f"{self.name}_recall0": float(recalls[0]),
                f"{self.name}_recall1": float(recalls[1]),
            }

        # Тройная классификация
        if n_classes == 3:
            t1, t2, mean_recall, recalls = self._find_threshold_trinary(
                list(y_true), list(y_score)
            )
            return {
                f"{self.name}_mean": float(mean_recall),
                f"{self.name}_threshold1": float(t1),
                f"{self.name}_threshold2": float(t2),
                f"{self.name}_recall0": float(recalls[0]),
                f"{self.name}_recall1": float(recalls[1]),
                f"{self.name}_recall2": float(recalls[2]),
            }

        # Для большего числа классов используем бинарный подход для каждой пары
        logger.warning(
            f"MeanRecallEvaluator supports 2 or 3 classes, got {n_classes}. Using binary approach for each pair."
        )
        results = {}
        for c1, c2 in itertools.combinations(unique_classes, 2):
            mask = np.isin(y_true, [c1, c2])
            if np.sum(mask) < 2:
                continue

            y_true_subset = y_true[mask]
            y_score_subset = y_score[mask]

            threshold, mean_recall, recalls = self._find_threshold_binary(
                list(y_true_subset), list(y_score_subset)
            )
            results[f"{self.name}_pair{c1}_{c2}_mean"] = float(mean_recall)
            results[f"{self.name}_pair{c1}_{c2}_threshold"] = float(threshold)

        if len(results) == 0:
            raise ValueError(f"Could not compute MeanRecall for {n_classes} classes")

        return results


class SequencialScoreEvaluator:
    def __init__(self, evaluators):
        self.evaluators = evaluators

    def __call__(self, similarity_scores, labels) -> Any:
        evaluation_res = {}
        for evaluator in self.evaluators:
            evaluation_res.update(evaluator(similarity_scores, labels))
        return evaluation_res
