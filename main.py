import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    dataset = instantiate(cfg.dataset)
    preprocessor = instantiate(cfg.preprocessor)
    dataset.apply_preprocessor(preprocessor)

    metric = instantiate(cfg.metric)
    metric_scores, gt_scores = metric.compute_scores(dataset)

    evaluator = instantiate(cfg.evaluator)

    results = evaluator(metric_scores, gt_scores)
    print("Evaluation results:", results)

if __name__ == "__main__":
    main()