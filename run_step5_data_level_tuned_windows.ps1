$ErrorActionPreference = "Stop"

if (-not (Test-Path ".\venv\Scripts\python.exe")) {
    throw "Virtual environment not found. Run setup_step1_windows.ps1 first."
}

& .\venv\Scripts\python.exe .\step5_run_data_level_fusion_tuned.py
