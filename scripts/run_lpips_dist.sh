#!/bin/bash
# LPIPS эксперименты с корреляциями (Spearman, Pearson, Kendall)

python main.py --multirun \
  dataset=paired_image_dist \
  evaluator=quality_corrs \
  metric=lpips,lpips_vgg,lpips_squeeze \
  preprocessor=lpips_basic,lpips_clahe,lpips_grayscale,lpips_hematoxylin,lpips_histogram_match,lpips_smoothed \
  'name=${hydra:runtime.choices.metric}_${hydra:runtime.choices.preprocessor}'
