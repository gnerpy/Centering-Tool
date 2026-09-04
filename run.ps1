# Start the centering bench.  Creates the virtual environment on first run.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
    & .\.venv\Scripts\python.exe -m pip install --quiet --upgrade pip
    & .\.venv\Scripts\python.exe -m pip install --quiet -r requirements.txt
}

# Scans are read from the parent folder unless BENCH_SCANS says otherwise.
& .\.venv\Scripts\python.exe server.py
