$ErrorActionPreference = "Stop"

Write-Host "[Step 1] UTD-MHAD environment setup (Windows)" -ForegroundColor Cyan

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python is not installed or not in PATH. Install Python 3.10+ and retry."
}

$pythonVersion = python --version
Write-Host "Detected $pythonVersion"

if (-not (Test-Path ".\venv")) {
    Write-Host "Creating virtual environment at .\\venv ..."
    python -m venv venv
} else {
    Write-Host "Virtual environment already exists at .\\venv"
}

$venvPython = ".\venv\Scripts\python.exe"
$venvPip = ".\venv\Scripts\pip.exe"

if (-not (Test-Path $venvPython)) {
    throw "venv Python executable not found at $venvPython"
}

Write-Host "Upgrading pip/setuptools/wheel..."
& $venvPython -m pip install --upgrade pip setuptools wheel

Write-Host "Installing PyTorch (CUDA 11.8 wheel index as in plan)..."
& $venvPip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Write-Host "Installing requirements.txt packages..."
& $venvPip install -r requirements.txt

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Activate venv with: .\\venv\\Scripts\\Activate.ps1"
