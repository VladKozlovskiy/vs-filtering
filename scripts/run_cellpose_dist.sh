#!/bin/bash
# CellPose эксперименты с корреляциями (Spearman, Pearson, Kendall)

python main.py --multirun \
  dataset=paired_image_dist \
  evaluator=quality_corrs \
  metric=cellpose \
  preprocessor=paired_cell_pipeline,cellpose_clahe,cellpose_grayscale,cellpose_hematoxylin,cellpose_histogram_match \
  'name=cellpose_${hydra:runtime.choices.preprocessor}'
