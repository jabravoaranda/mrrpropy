param(
    [int[]]$Years = @(2025, 2024, 2023),
    [int]$MaxParallel = 8,
    [string]$RawBase = "Z:\UGR\mrrpro81",
    [string]$ProductsBase = "W:\mrrpropy_products",
    [string]$ScratchBase = "",
    [switch]$SkipAggregate
)

$ErrorActionPreference = "Stop"

$Repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Python = Join-Path $Repo ".venv\Scripts\python.exe"
$Script = Join-Path $Repo "workbench\scripts\analyze_bimonthly_process_stats.py"

if (-not $ScratchBase) {
    $ScratchBase = Join-Path $Repo "workbench\output"
}

function Wait-CampaignSlot {
    param([System.Collections.ArrayList]$Jobs, [int]$MaxParallel)

    while ($Jobs.Count -ge $MaxParallel) {
        for ($i = $Jobs.Count - 1; $i -ge 0; $i--) {
            $job = $Jobs[$i]
            $proc = Get-Process -Id $job.ProcessId -ErrorAction SilentlyContinue
            if ($null -eq $proc) {
                $exit = $job.Process.ExitCode
                if ($exit -ne 0) {
                    throw "Job failed with exit code $exit. See $($job.StdErr)"
                }
                $Jobs.RemoveAt($i)
            }
        }
        if ($Jobs.Count -ge $MaxParallel) {
            Start-Sleep -Seconds 30
        }
    }
}

function Start-ProcessMonth {
    param(
        [int]$Year,
        [int]$Month,
        [System.Collections.ArrayList]$Jobs
    )

    $start = Get-Date -Year $Year -Month $Month -Day 1 -Hour 0 -Minute 0 -Second 0
    $end = $start.AddMonths(1)
    $ym = "{0:0000}_{1:00}" -f $Year, $Month
    $yearName = "process_campaign_$Year"

    $rawRoot = Join-Path $RawBase "$Year"
    $outputDir = Join-Path $ScratchBase "$yearName\$ym"
    $productDir = Join-Path $outputDir "products"
    $archiveProductDir = Join-Path $ProductsBase "$yearName\products"
    $logDir = Join-Path $ScratchBase "$yearName\logs"

    New-Item -ItemType Directory -Force -Path $outputDir, $productDir, $archiveProductDir, $logDir | Out-Null

    $stdout = Join-Path $logDir "$ym.stdout.log"
    $stderr = Join-Path $logDir "$ym.stderr.log"
    $arguments = @(
        "-u", "`"$Script`"",
        "--raw-root", "`"$rawRoot`"",
        "--start-date", $start.ToString("yyyy-MM-dd"),
        "--end-date", $end.ToString("yyyy-MM-dd"),
        "--output-dir", "`"$outputDir`"",
        "--product-dir", "`"$productDir`"",
        "--archive-product-dir", "`"$archiveProductDir`"",
        "--clean-local-products",
        "--window-thickness-m", "500",
        "--min-tau-strength", "0.3",
        "--dpi", "300",
        "--process-only"
    ) -join " "

    Write-Host "[launch] $ym"
    $process = Start-Process -FilePath $Python -ArgumentList $arguments -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
    [void]$Jobs.Add([pscustomobject]@{
        Year = $Year
        Month = $Month
        ProcessId = $process.Id
        Process = $process
        StdOut = $stdout
        StdErr = $stderr
    })
}

foreach ($year in $Years) {
    $jobs = [System.Collections.ArrayList]::new()
    for ($month = 1; $month -le 12; $month++) {
        Wait-CampaignSlot -Jobs $jobs -MaxParallel $MaxParallel
        Start-ProcessMonth -Year $year -Month $month -Jobs $jobs
    }

    while ($jobs.Count -gt 0) {
        Wait-CampaignSlot -Jobs $jobs -MaxParallel 1
    }

    if (-not $SkipAggregate) {
        $yearName = "process_campaign_$year"
        $rawRoot = Join-Path $RawBase "$year"
        $outputDir = Join-Path $ProductsBase "$yearName"
        $productDir = Join-Path $outputDir "products"
        New-Item -ItemType Directory -Force -Path $outputDir, $productDir | Out-Null

        Write-Host "[aggregate] $year"
        & $Python -u $Script `
            --raw-root $rawRoot `
            --start-date "$year-01-01" `
            --end-date "$($year + 1)-01-01" `
            --output-dir $outputDir `
            --product-dir $productDir `
            --window-thickness-m 500 `
            --min-tau-strength 0.3 `
            --dpi 300 `
            --aggregate-only

        if ($LASTEXITCODE -ne 0) {
            throw "Aggregation failed for $year with exit code $LASTEXITCODE"
        }
    }
}
