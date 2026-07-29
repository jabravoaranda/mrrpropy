# Paper Figure Workflow

`Workbench/scripts/build_paper_figures.py` creates the paper-ready figures from
MRR-PRO data. The editable defaults are kept in
`Workbench/scripts/paper_figures_config.py`, similar to configuration cells in
the full processing workflow.

## 1. Configure the run

Edit `Workbench/scripts/paper_figures_config.py` before a long run. The most
important settings are:

```python
RAW_DIR = Path(r"C:\path\to\raw\files")
OUTPUT_DIR = Path("Workbench/output/paper_figures")
PRODUCT_DIR = OUTPUT_DIR / "products"

K = 11
WINDOW_THICKNESS_M = 500.0
WINDOW_STEP_M = None
MIN_TAU_STRENGTH = 0.3

# Omit process labels that are not useful for the paper.
EXCLUDE_PROCESSES = ("steady_or_weak",)

# Approximate KDE display domains.
KDE_DOMAIN_SIGMA = 3.0
KDE_VARIABLE_LIMITS = {"LWC": (0.0, 0.3)}
```

Command-line arguments override the values in this file.

## 2. Process raw files

Run the script from the repository root:

```powershell
.venv\Scripts\python.exe Workbench\scripts\build_paper_figures.py
```

For testing one or more raw files, point `RAW_DIR` at a small test folder or
pass a folder explicitly:

```powershell
.venv\Scripts\python.exe Workbench\scripts\build_paper_figures.py `
  --raw-dir "Workbench/test files/qtest"
```

The run stores processed NetCDF products and completed sliding-process CSV
checkpoints in `PRODUCT_DIR`. The per-file sliding CSVs are the durable
classification results used for aggregate process plots.

When per-file figure generation is enabled, figure 09 is the processed-only
quicklook. It uses the processed `Ze` field, the same reflectivity scale and
height limits as figure 01, and no colorbar.

To add per-file graph 10, set the ranges in metres for the
droplet-size-distribution curves:

```python
DSD_RANGES_M = [1500.0, 2000.0, 2700.0]
```

or pass it on the command line:

```powershell
.venv\Scripts\python.exe Workbench\scripts\build_paper_figures.py `
  --raw-dir "Workbench/test files/qtest" `
  --dsd-ranges-m 1500 2000 2700
```

Graph 10 uses the existing `MRRProData.plot_DSD_by_range` function with a
list of requested ranges and selects the nearest available range gate for each
value. Existing cached products are not rebuilt automatically; if a cached
product lacks `dsd_3D`, graph 10 is skipped with a warning. If graph 10 is
requested while products are newly created or explicitly rebuilt with
`--reset-cache 1`, the paper workflow saves `dsd_3D` in those products.

To rebuild cached processed products and sliding scans while testing:

```powershell
.venv\Scripts\python.exe Workbench\scripts\build_paper_figures.py `
  --raw-dir "Workbench/test files/qtest" `
  --reset-cache 1
```

Use `--reset-cache 1` deliberately because it can repeat the long processing
step. A value of `0` reuses existing products and sliding CSVs.

## 3. Aggregate completed sliding scans

When many files have already been processed, aggregate the completed sliding
CSV checkpoints without reopening or reprocessing the raw files:

```powershell
.venv\Scripts\python.exe Workbench\scripts\build_paper_figures.py `
  --aggregate-products
```

The script recursively finds files matching:

```text
*_raprompro_sliding.csv
```

It streams all rows into:

```text
Workbench/output/paper_figures/combined_column_process_scan.csv
```

For the xarray and KDE calculations, it retains a bounded sample per process
to keep memory use reasonable. The default is 50,000 rows per process and can
be changed with:

```powershell
.venv\Scripts\python.exe Workbench\scripts\build_paper_figures.py `
  --aggregate-products `
  --aggregate-sample-per-process 100000 `
  --aggregate-chunksize 100000
```

The individual sliding CSVs are the progress checkpoints. If aggregation or
raw processing is interrupted, run the same command again; completed product
files remain available and are reused.

## 4. Generated aggregate figures

The aggregate run creates:

```text
Workbench/output/paper_figures/kde_Dw_V_LWC_N_by_process.png
Workbench/output/paper_figures/kde_bb_distance_delta_v_by_process.png
Workbench/output/paper_figures/combined_process_samples.nc
```

The KDE figures use the process colors defined by the paper plotting module.
`steady_or_weak` is excluded by default. To include all process labels for a
run, provide an empty exclusion list at the end of the command:

```powershell
.venv\Scripts\python.exe Workbench\scripts\build_paper_figures.py `
  --aggregate-products `
  --exclude-processes
```

To keep only a selected set of processes:

```powershell
.venv\Scripts\python.exe Workbench\scripts\build_paper_figures.py `
  --aggregate-products `
  --processes activation growth_depletion_gain growth_depletion_loss evaporation
```

## 5. Replot an existing combined CSV

If `combined_column_process_scan.csv` already exists and only the aggregate
figures need to be regenerated, use it as a scan input:

```powershell
.venv\Scripts\python.exe Workbench\scripts\build_paper_figures.py `
  --scan-glob "Workbench/output/paper_figures/combined_column_process_scan.csv"
```

For a large collection, `--aggregate-products` is preferred because it reads
the product checkpoints in chunks and limits the in-memory KDE sample.

## 6. NetCDF versus CSV products

The stored `*_raprompro.nc` files are full processed MRR-PRO products. They do
not contain the flattened sliding-column process table by themselves. The
`*_raprompro_sliding.csv` files contain the completed sliding classifications,
including process labels, Kendall/Thiel scores, bright-band distance, and
velocity differences.

For this paper workflow:

- Use processed NetCDF files to generate or rebuild missing sliding scans.
- Use completed sliding CSV files for fast multi-file aggregation.
- Keep both formats; NetCDF is the scientific source product, while the
  sliding CSV is the convenient table checkpoint for process statistics.

## 7. Height limits and profile domains

The paper plotting functions use height limits relative to the bright band.
The full-range figures use the bright-band top, while the below-bright-band
figures use the bright-band peak and an upper offset. The lower bound is
normally 700 m.

The MPP profile and tau/Thiel profile panels automatically scale each x-axis
from the finite data visible within the selected height range. This prevents a
small number of excluded upper points from expanding the LWC, Dm, or Nw
domains.

## 8. Output location

All aggregate figures and tables are written under `OUTPUT_DIR`. Per-file
figures, when enabled in the script, are written into a subdirectory named
after each raw file beneath the same output directory.
