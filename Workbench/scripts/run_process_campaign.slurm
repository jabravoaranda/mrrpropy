#!/usr/bin/env bash
#SBATCH --job-name=mrrpropy-process
#SBATCH --array=0-35
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/mrrpropy_%A_%a.out
#SBATCH --error=logs/mrrpropy_%A_%a.err

set -euo pipefail

# Adapt these paths to the HPC mount points.
REPO="${REPO:-$HOME/mrrpropy}"
RAW_BASE="${RAW_BASE:-/path/to/RAW/UGR/mrrpro81}"
PRODUCTS_BASE="${PRODUCTS_BASE:-/path/to/PRODUCTS/mrrpropy_products}"
SCRATCH_BASE="${SCRATCH_BASE:-${TMPDIR:-$HOME/scratch}/mrrpropy}"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"

YEARS=(2025 2024 2023)
YEAR_INDEX=$((SLURM_ARRAY_TASK_ID / 12))
MONTH=$((SLURM_ARRAY_TASK_ID % 12 + 1))
YEAR="${YEARS[$YEAR_INDEX]}"

START=$(printf "%04d-%02d-01" "$YEAR" "$MONTH")
if [[ "$MONTH" -eq 12 ]]; then
  END=$(printf "%04d-01-01" "$((YEAR + 1))")
else
  END=$(printf "%04d-%02d-01" "$YEAR" "$((MONTH + 1))")
fi
YM=$(printf "%04d_%02d" "$YEAR" "$MONTH")

RAW_ROOT="$RAW_BASE/$YEAR"
OUTPUT_DIR="$SCRATCH_BASE/process_campaign_$YEAR/$YM"
PRODUCT_DIR="$OUTPUT_DIR/products"
ARCHIVE_PRODUCT_DIR="$PRODUCTS_BASE/process_campaign_$YEAR/products"

mkdir -p "$OUTPUT_DIR" "$PRODUCT_DIR" "$ARCHIVE_PRODUCT_DIR" logs

"$PYTHON" -u "$REPO/workbench/scripts/analyze_bimonthly_process_stats.py" \
  --raw-root "$RAW_ROOT" \
  --start-date "$START" \
  --end-date "$END" \
  --output-dir "$OUTPUT_DIR" \
  --product-dir "$PRODUCT_DIR" \
  --archive-product-dir "$ARCHIVE_PRODUCT_DIR" \
  --clean-local-products \
  --window-thickness-m 500 \
  --min-tau-strength 0.3 \
  --dpi 300 \
  --process-only
