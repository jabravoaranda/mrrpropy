#!/usr/bin/env bash
#SBATCH --job-name=mrrpropy-aggregate
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/mrrpropy_aggregate_%j.out
#SBATCH --error=logs/mrrpropy_aggregate_%j.err

set -euo pipefail

YEAR="${YEAR:?Set YEAR=2025, 2024, or 2023 when submitting.}"
REPO="${REPO:-$HOME/mrrpropy}"
RAW_BASE="${RAW_BASE:-/path/to/RAW/UGR/mrrpro81}"
PRODUCTS_BASE="${PRODUCTS_BASE:-/path/to/PRODUCTS/mrrpropy_products}"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"

RAW_ROOT="$RAW_BASE/$YEAR"
OUTPUT_DIR="$PRODUCTS_BASE/process_campaign_$YEAR"
PRODUCT_DIR="$OUTPUT_DIR/products"
END_YEAR=$((YEAR + 1))

mkdir -p "$OUTPUT_DIR" "$PRODUCT_DIR" logs

"$PYTHON" -u "$REPO/workbench/scripts/analyze_bimonthly_process_stats.py" \
  --raw-root "$RAW_ROOT" \
  --start-date "$YEAR-01-01" \
  --end-date "$END_YEAR-01-01" \
  --output-dir "$OUTPUT_DIR" \
  --product-dir "$PRODUCT_DIR" \
  --window-thickness-m 500 \
  --min-tau-strength 0.3 \
  --dpi 300 \
  --aggregate-only
