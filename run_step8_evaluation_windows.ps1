$ErrorActionPreference = "Stop"

if (-not (Test-Path ".\venv\Scripts\python.exe")) {
    throw "Virtual environment not found. Run setup_step1_windows.ps1 first."
}

& .\venv\Scripts\python.exe .\step8_generate_evaluation_report.py
