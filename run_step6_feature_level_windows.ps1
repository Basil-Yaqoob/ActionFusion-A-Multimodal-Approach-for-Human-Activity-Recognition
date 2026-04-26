$ErrorActionPreference = "Stop"

if (-not (Test-Path ".\venv\Scripts\python.exe")) {
    throw "Virtual environment not found. Run setup_step1_windows.ps1 first."
}

& .\venv\Scripts\python.exe .\step6_run_feature_level_fusion.py
