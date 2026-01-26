#!/bin/bash
# Verification of old code configurations (correlation metrics)
# Expected results (Spearman):
# 1. squeeze_avg_rgb: 0.646
# 2. alex_avg_rgb:    0.623
# 3. squeeze_avg_gray: 0.610
# 4. vgg_avg_rgb:     0.597
# 5. vgg_lin_rgb:     0.590

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "=== Old Code Config 1: squeeze_avg_rgb ==="
python main.py \
  dataset=paired_image_dist \
  evaluator=quality_corrs \
  metric=lpips_squeeze_avg \
  preprocessor=old_code_1_squeeze_avg_rgb \
  name=old_code_1_squeeze_avg_rgb

echo "=== Old Code Config 2: alex_avg_rgb ==="
python main.py \
  dataset=paired_image_dist \
  evaluator=quality_corrs \
  metric=lpips_alex_avg \
  preprocessor=old_code_2_alex_avg_rgb \
  name=old_code_2_alex_avg_rgb

echo "=== Old Code Config 3: squeeze_avg_gray ==="
python main.py \
  dataset=paired_image_dist \
  evaluator=quality_corrs \
  metric=lpips_squeeze_avg \
  preprocessor=old_code_3_squeeze_avg_gray \
  name=old_code_3_squeeze_avg_gray

echo "=== Old Code Config 4: vgg_avg_rgb ==="
python main.py \
  dataset=paired_image_dist \
  evaluator=quality_corrs \
  metric=lpips_vgg_avg \
  preprocessor=old_code_4_vgg_avg_rgb \
  name=old_code_4_vgg_avg_rgb

echo "=== Old Code Config 5: vgg_lin_rgb ==="
python main.py \
  dataset=paired_image_dist \
  evaluator=quality_corrs \
  metric=lpips_vgg_lin \
  preprocessor=old_code_5_vgg_lin_rgb \
  name=old_code_5_vgg_lin_rgb

echo "=== DONE ==="
