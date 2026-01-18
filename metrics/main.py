"""
Модуль для оценки схожести парных изображений.
"""
from abc import ABC, abstractmethod
from typing import Iterable, List, Any, Callable, Optional, Tuple, Dict
import torch
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
from tqdm import tqdm 
from lpips import LPIPS
from cellpose import models, core, io, plot
import torch.nn.functional as F
import logging
import numpy as np

logger = logging.getLogger(__name__)


class BaseSimilarityEstimator(ABC):

    @abstractmethod
    def compute_scores(self, dataset: Dataset) -> Dict[str, List[float]]:
        pass

class LambdaSimilarityEstimator(BaseSimilarityEstimator):

    def __init__(self, fn: Callable[[Any, Any], float], name: str):
        self.fn = fn
        self.name = name
    
    def compute_scores(self, dataset: Dataset) -> Dict[str, List[float]]:
        scores = []
        for f_patch, w_patch, dist, target in tqdm(dataset, desc=self.name):
            try:
                score = self.fn(f_patch, w_patch)
                scores.append(float(score))
            except Exception as e:
                logger.error(f"Error computing similarity for sample: {e}")
                raise
        return {self.name: scores}

class LPIPSSimilarityEstimator(BaseSimilarityEstimator):

    def __init__(self, lpips_args: Dict[str, Any] = None, device: str = 'cuda'): 
        if lpips_args is None:
            lpips_args = {}
        self.device = device
        self.model = LPIPS(**lpips_args).to(device)
        self.name = 'LPIPS'
        logger.info(f"Initialized LPIPS model on {device}")
        
    def compute_scores(self, dataset: Dataset) -> Dict[str, List[float]]:
        """Вычисляет LPIPS метрики схожести."""
        scores = []
        for f_patch, w_patch, dist, target in tqdm(dataset, desc=self.name):
            try:
                # Конвертируем в тензоры если нужно
                if not isinstance(f_patch, torch.Tensor):
                    f_patch = torch.from_numpy(f_patch) if isinstance(f_patch, np.ndarray) else torch.tensor(f_patch)
                if not isinstance(w_patch, torch.Tensor):
                    w_patch = torch.from_numpy(w_patch) if isinstance(w_patch, np.ndarray) else torch.tensor(w_patch)
                
                # Добавляем batch dimension и перемещаем на устройство
                if f_patch.ndim == 3:
                    f_patch = f_patch.unsqueeze(0)
                if w_patch.ndim == 3:
                    w_patch = w_patch.unsqueeze(0)
                
                f_patch = f_patch.to(self.device)
                w_patch = w_patch.to(self.device)
                
                with torch.inference_mode():
                    score = self.model(f_patch, w_patch).mean().item()
                scores.append(score)
            except Exception as e:
                logger.error(f"Error computing LPIPS for sample: {e}")
                raise
        return {self.name: scores}

class CellPoseSimilarityEstimator(BaseSimilarityEstimator):

    def __init__(self, device: str = 'cuda', batch_size: int = 16): 
        self.device = device
        self.batch_size = batch_size
        try:
            self.model = models.CellposeModel(gpu=(device == 'cuda'), pretrained_model='cyto2_cp3')
            self.name = 'CellPoseSAM'
            logger.info(f"Initialized CellPose model on {device} with batch_size={batch_size}")
        except Exception as e:
            logger.error(f"Failed to initialize CellPose model: {e}")
            raise


    @staticmethod
    def cosine_sim(im1, im2):
        im1= im1.reshape(im1.shape[0], -1) 
        im2 = im2.reshape(im2.shape[0], -1)
        return F.cosine_similarity(im1, im2, dim=1)

    @staticmethod
    def normalized_dist(im1, im2):
        im1 = im1.reshape(im1.shape[0], -1)
        im2 = im2.reshape(im2.shape[0], -1)
        im1 = F.normalize(im1, dim=1)
        im2 = F.normalize(im2, dim=1)
        return torch.norm(im1 - im2, dim=1)
    
    def compute_scores(self, dataset: Dataset) -> Dict[str, List[float]]:
        encoder = self.model.net.encoder
        results = defaultdict(list)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        
        try:
            with torch.inference_mode():
                for batch_idx, batch in enumerate(tqdm(loader, desc=self.name)):
                    try:
                        batch_he, batch_ihc, dist, target = batch
                        
                        # Конвертируем в тензоры и перемещаем на устройство
                        if not isinstance(batch_he, torch.Tensor):
                            batch_he = torch.stack([torch.from_numpy(x) if isinstance(x, np.ndarray) else torch.tensor(x) for x in batch_he])
                        if not isinstance(batch_ihc, torch.Tensor):
                            batch_ihc = torch.stack([torch.from_numpy(x) if isinstance(x, np.ndarray) else torch.tensor(x) for x in batch_ihc])
                        
                        # Нормализуем если нужно (ожидается [0, 255] или [0, 1])
                        if batch_he.max() > 1.0:
                            batch_he = batch_he / 255.0
                        if batch_ihc.max() > 1.0:
                            batch_ihc = batch_ihc / 255.0
                        
                        in0 = batch_he.to(self.device).to(torch.bfloat16)
                        in1 = batch_ihc.to(self.device).to(torch.bfloat16)
                        
                        x0 = encoder.patch_embed(in0)
                        x1 = encoder.patch_embed(in1)
                        
                        # Проходим по слоям энкодера
                        for i, blk in enumerate(encoder.blocks):
                            x0 = blk(x0)
                            x1 = blk(x1)
                
                            sim = self.cosine_sim(x0, x1).float().cpu().numpy()
                            dist_metric = self.normalized_dist(x0, x1).float().cpu().numpy()
                            
                            results[f"{self.name}_sim_layer_{i}"].extend(sim.tolist())
                            results[f"{self.name}_dist_layer_{i}"].extend(dist_metric.tolist())
                
                        # Финальный слой (neck)
                        x0_neck = encoder(in0)
                        x1_neck = encoder(in1)
                
                        results[f"{self.name}_sim_neck"].extend(
                            self.cosine_sim(x0_neck, x1_neck).float().cpu().numpy().tolist()
                        )
                        results[f"{self.name}_dist_neck"].extend(
                            self.normalized_dist(x0_neck, x1_neck).float().cpu().numpy().tolist()
                        )
                    except Exception as e:
                        logger.error(f"Error processing batch {batch_idx}: {e}")
                        raise
        except Exception as e:
            logger.error(f"Error in compute_scores: {e}")
            raise
        
        logger.info(f"Computed {len(results)} metrics for {len(list(results.values())[0])} samples")
        return dict(results)