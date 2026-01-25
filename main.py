import logging
from pathlib import Path
from pprint import pprint

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="cellpose_dist")
def main(cfg: DictConfig) -> None:
    seed = cfg.get("seed", 42)
    seed_everything(seed, workers=True)

    logger.info("Experiment configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    logger.info(f"Output directory: {output_dir}")

    dataset = instantiate(cfg.dataset)
    logger.info(f"Loaded dataset with {len(dataset)} samples")

    preprocessor = instantiate(cfg.preprocessor)
    dataset.apply_preprocessor(preprocessor)
    logger.info("Preprocessor applied")

    metric = instantiate(cfg.metric)
    logger.info(f"Using metric: {metric.__class__.__name__}")

    logger.info("Computing similarity scores...")
    predicted_similarities = metric.compute_scores(dataset)
    logger.info(f"Computed similarities for {len(predicted_similarities)} metrics")

    gt_similarities = [item for _, _, item in dataset]
    logger.info(f"Collected {len(gt_similarities)} ground truth labels")

    evaluator = instantiate(cfg.evaluator)
    logger.info(f"Using evaluator: {evaluator.__class__.__name__}")

    results = evaluator(predicted_similarities, gt_similarities)
    logger.info("Evaluation results:")
    pprint(results)

    results_path = output_dir / "results.yaml"
    OmegaConf.save(results, results_path)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
