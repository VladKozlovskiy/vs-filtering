from abc import ABC, abstractmethod
from typing import Iterable, List, Any, Callable, Optional, Tuple
import torch
from collections import defaultdict
from tqdm import tqdm 
from lpips import LPIPS
from cellpose import models, core, io, plot
import torch.nn.functional as F

class BaseSimilarityEstimator(ABC):
    @abstractmethod
    def compute_scores(self, dataset) -> Any:
        pass

class LambdaSimilarityEstimator(BaseSimilarityEstimator):
    def __init__(self, fn: Callable[[Any], Any], name):
        self.fn = fn
        self.name = name
    def compute_scores(self, dataset) -> Any:
        scores = []
        for f_patch, w_patch, dist, target in tqdm(dataset):
            self.fn(f_patch, w_patch)
        return { self.name : scores }

class LPIPSSimilarityEstimator(BaseSimilarityEstimator):
    def __init__(self, lpips_args): 
        self.model = LPIPS(**lpips_args).to('cuda')
        self.name = 'LPIPS'
        
    def compute_scores(self, dataset) -> Any:
        scores = []
        for f_patch, w_patch, dist, target in tqdm(dataset, desc=self.name):
            with torch.inference_mode():
                scores.append( lpips_model(f_patch, w_patch).mean().item() ) 
        return { self.name : scores }

class CellPoseSimilarityEstimator(BaseSimilarityEstimator):
    def __init__(self): 
        self.device = 'gpu'
        self.model = models.CellposeModel(gpu=True, pretrained_model='cyto2_cp3').to(self.device)
        self.name = 'CellPoseSAM'
        
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
    
    def compute_scores(self, dataset) -> Any:
        encoder = self.model.net.encoder
        results = defaultdict(list)
        with torch.inference_mode():
            # Переписать на батчевую обработку 
            for batch_he, batch_ihc, dist, target in tqdm(dataset):
                in0 = batch_he.to(device)
                in1 = batch_ihc.to(device)
                
                x0 = encoder.patch_embed(in0)
                x1 = encoder.patch_embed(in1)
                
                for i, blk in enumerate(encoder.blocks):
                    x0 = blk(x0)
                    x1 = blk(x1)
        
                    sim = self.cosine_sim(x0, x1).float().cpu().numpy()
                    dist = self.normalized_dist(x0, x1).float().cpu().numpy()
                    
                    results[f"{self.names}_sim_layer_{i}"].extend(sim)
                    results[f"{self.names}_dist_layer_{i}"].extend(dist)
        
        
                x0_neck = encoder(in0)
                x1_neck = encoder(in1)
        
                results[f"{self.names}_sim_neck"].extend(
                    cosine_sim(x0_neck, x1_neck).float().cpu().numpy()
                )
                results[f"{self.names}_dist_neck"].extend(
                    normalized_dist(x0_neck, x1_neck).float().cpu().numpy()
                )
        return results