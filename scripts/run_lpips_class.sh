#!/bin/bash
# LPIPS эксперименты с классификацией (AUC, MeanRecall)

python main.py --multirun \
  dataset=paired_image_class \
  evaluator=quality_auc_recall \
  metric=lpips,lpips_vgg,lpips_squeeze \
  preprocessor=lpips_basic,lpips_clahe,lpips_grayscale,lpips_hematoxylin,lpips_histogram_match,lpips_smoothed \
  'name=${hydra:runtime.choices.metric}_${hydra:runtime.choices.preprocessor}'
