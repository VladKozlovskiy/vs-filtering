#!/bin/bash
# Запуск всех экспериментов

set -e
cd "$(dirname "$0")/.."

echo "=== LPIPS DIST ==="
bash scripts/run_lpips_dist.sh

echo "=== LPIPS CLASS ==="
bash scripts/run_lpips_class.sh

echo "=== CELLPOSE DIST ==="
bash scripts/run_cellpose_dist.sh

echo "=== CELLPOSE CLASS ==="
bash scripts/run_cellpose_class.sh

echo "=== DONE ==="
