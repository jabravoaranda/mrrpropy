# Rain-process campaign processing

This workflow processes MRR-PRO RAW files into re-entrant hourly checkpoints and
year-level aggregate figures.

## Canonical process labels

New outputs use the canonical labels from
`mrrpropy.rain_process_classification.rain_process_info.PROCESS_SIGNATURES`.
Legacy labels remain readable through aliases:

| Legacy label | Canonical label |
| --- | --- |
| `growth_depletion` | `coalescence` |
| `growth_depletion_gain` | `coalescence_gain` |
| `growth_depletion_loss` | `coalescence_loss` |
| `condensation` | `activation` |
| `evaporation` | `evaporation_strong` |

The alias layer is only for input compatibility. It prevents old checkpoints
from becoming separate classes during aggregation.

## Local Windows campaign

Run from the repository root:

```powershell
.\workbench\scripts\run_process_campaign.ps1 `
  -Years 2025,2024,2023 `
  -MaxParallel 8 `
  -RawBase "Z:\UGR\mrrpro81" `
  -ProductsBase "W:\mrrpropy_products"
```

The script writes temporary products under `Workbench/output`, archives durable
checkpoints under `W:\mrrpropy_products`, and removes verified local products to
keep `C:` from filling.

## SLURM campaign

Copy or clone the repository on the HPC, create the Python environment, then
submit the monthly array:

```bash
mkdir -p logs
sbatch \
  --export=ALL,REPO="$HOME/mrrpropy",RAW_BASE="/path/to/RAW/UGR/mrrpro81",PRODUCTS_BASE="/path/to/PRODUCTS/mrrpropy_products" \
  workbench/scripts/run_process_campaign.slurm
```

The array has 36 tasks: all months in 2025, then 2024, then 2023. Each task is
re-entrant: if the archived `*_raprompro_sliding.csv` exists, it is loaded
instead of recalculated.

Aggregate a completed year with:

```bash
sbatch \
  --export=ALL,YEAR=2025,REPO="$HOME/mrrpropy",RAW_BASE="/path/to/RAW/UGR/mrrpro81",PRODUCTS_BASE="/path/to/PRODUCTS/mrrpropy_products" \
  workbench/scripts/aggregate_process_campaign_year.slurm
```
