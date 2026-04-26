$ErrorActionPreference = "Stop"

if (-not (Test-Path ".\venv\Scripts\python.exe")) {
    throw "Virtual environment not found. Run setup_step1_windows.ps1 first."
}

# Remove --max-samples to preprocess the full dataset.
& .\venv\Scripts\python.exe .\step3_preprocess_and_cache.py --max-samples 25
