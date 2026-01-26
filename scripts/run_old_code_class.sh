#!/bin/bash
# Verification of old code configurations (classification metrics)
# Expected results (AUC):
# 1. squeeze_avg_rgb: 0.796
# 2. alex_avg_rgb:    0.796
# 3. squeeze_avg_gray: 0.784
# 4. vgg_avg_rgb:     0.788
# 5. vgg_lin_rgb:     0.791

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "=== Old Code Config 1: squeeze_avg_rgb (class) ==="
python main.py \
  dataset=paired_image_class \
  evaluator=quality_auc_recall \
  metric=lpips_squeeze_avg \
  preprocessor=old_code_1_squeeze_avg_rgb \
  name=old_code_1_squeeze_avg_rgb_class

echo "=== Old Code Config 2: alex_avg_rgb (class) ==="
python main.py \
  dataset=paired_image_class \
  evaluator=quality_auc_recall \
  metric=lpips_alex_avg \
  preprocessor=old_code_2_alex_avg_rgb \
  name=old_code_2_alex_avg_rgb_class

echo "=== Old Code Config 3: squeeze_avg_gray (class) ==="
python main.py \
  dataset=paired_image_class \
  evaluator=quality_auc_recall \
  metric=lpips_squeeze_avg \
  preprocessor=old_code_3_squeeze_avg_gray \
  name=old_code_3_squeeze_avg_gray_class

echo "=== Old Code Config 4: vgg_avg_rgb (class) ==="
python main.py \
  dataset=paired_image_class \
  evaluator=quality_auc_recall \
  metric=lpips_vgg_avg \
  preprocessor=old_code_4_vgg_avg_rgb \
  name=old_code_4_vgg_avg_rgb_class

echo "=== Old Code Config 5: vgg_lin_rgb (class) ==="
python main.py \
  dataset=paired_image_class \
  evaluator=quality_auc_recall \
  metric=lpips_vgg_lin \
  preprocessor=old_code_5_vgg_lin_rgb \
  name=old_code_5_vgg_lin_rgb_class

echo "=== DONE ==="
