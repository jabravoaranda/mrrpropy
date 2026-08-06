$ErrorActionPreference = "Stop"

$Repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Python = Join-Path $Repo ".venv\Scripts\python.exe"
$Script = Join-Path $Repo "workbench\scripts\analyze_bimonthly_process_stats.py"
$OutputDir = Join-Path $Repo "workbench\output\bimonthly_process_stats_2025_03_04"
$ProductDir = Join-Path $OutputDir "products"

New-Item -ItemType Directory -Force -Path $OutputDir, $ProductDir | Out-Null

function Invoke-ProcessStatsMonth {
    param(
        [Parameter(Mandatory = $true)][string]$Month,
        [Parameter(Mandatory = $true)][string]$StartDate,
        [Parameter(Mandatory = $true)][string]$EndDate
    )

    & $Python -u $Script `
        --raw-root "Z:\UGR\mrrpro81\2025\$Month" `
        --start-date $StartDate `
        --end-date $EndDate `
        --output-dir $OutputDir `
        --product-dir $ProductDir `
        --window-thickness-m 500 `
        --min-tau-strength 0.3 `
        --dpi 300

    if ($LASTEXITCODE -ne 0) {
        throw "Month $Month failed with exit code $LASTEXITCODE"
    }
}

Invoke-ProcessStatsMonth -Month "03" -StartDate "2025-03-01" -EndDate "2025-04-01"
Invoke-ProcessStatsMonth -Month "04" -StartDate "2025-04-01" -EndDate "2025-05-01"
