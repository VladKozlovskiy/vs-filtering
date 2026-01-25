#!/bin/bash
# CellPose эксперименты с классификацией (AUC, MeanRecall)

python main.py --multirun \
  dataset=paired_image_class \
  evaluator=quality_auc_recall \
  metric=cellpose \
  preprocessor=paired_cell_pipeline,cellpose_clahe,cellpose_grayscale,cellpose_hematoxylin,cellpose_histogram_match \
  'name=cellpose_${hydra:runtime.choices.preprocessor}'
