<#
  Bootstrap script for local development on Windows.
  Responsibilities:
  - Validate/install required tools (Git, Python, Node.js, Docker Desktop)
  - Prepare backend/frontend environment files
  - Optionally inject OPENAI_API_KEY into backend/.env
  - Install backend/frontend dependencies and run smoke checks
#>

param(
    [string]$OpenAIApiKey = "",
    [switch]$SkipInstall,
    [switch]$ReconfigureOnly,
    [switch]$ForceEnvReset
)

$ErrorActionPreference = "Stop"

function Write-Step([string]$Message) {
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Ensure-Command([string]$Name) {
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Ensure-WingetPackage([string]$CommandName, [string]$PackageId, [string]$FriendlyName) {
    if (Ensure-Command $CommandName) {
        Write-Host "$FriendlyName already installed." -ForegroundColor Green
        return
    }
    if ($SkipInstall) {
        throw "$FriendlyName is missing. Re-run without -SkipInstall to auto-install."
    }
    if (-not (Ensure-Command "winget")) {
        throw "winget is not available. Install App Installer from Microsoft Store and retry."
    }
    Write-Host "Installing $FriendlyName via winget..." -ForegroundColor Yellow
    winget install --id $PackageId --exact --accept-source-agreements --accept-package-agreements
    if (-not (Ensure-Command $CommandName)) {
        throw "$FriendlyName installation finished but command '$CommandName' is still unavailable. Open a new terminal and rerun."
    }
}

function Ensure-EnvFile([string]$ExamplePath, [string]$TargetPath) {
    if ($ForceEnvReset -and (Test-Path $TargetPath)) {
        Copy-Item $ExamplePath $TargetPath -Force
        Write-Host "Reset $TargetPath from template." -ForegroundColor Yellow
        return
    }
    if (-not (Test-Path $TargetPath)) {
        Copy-Item $ExamplePath $TargetPath
        Write-Host "Created $TargetPath from template." -ForegroundColor Green
    } else {
        Write-Host "$TargetPath already exists. Keeping current file." -ForegroundColor Yellow
    }
}

if (-not $ReconfigureOnly) {
    Write-Step "Checking required tools"
    Ensure-WingetPackage -CommandName "git" -PackageId "Git.Git" -FriendlyName "Git"
    Ensure-WingetPackage -CommandName "python" -PackageId "Python.Python.3.13" -FriendlyName "Python"
    Ensure-WingetPackage -CommandName "node" -PackageId "OpenJS.NodeJS.LTS" -FriendlyName "Node.js"
    Ensure-WingetPackage -CommandName "docker" -PackageId "Docker.DockerDesktop" -FriendlyName "Docker Desktop"

    Write-Step "Version validation"
    git --version
    python --version
    node --version
    npm --version
    docker --version
    docker compose version
}

Write-Step "Creating project env files"
Ensure-EnvFile -ExamplePath "backend/.env.example" -TargetPath "backend/.env"
Ensure-EnvFile -ExamplePath "frontend/.env.example" -TargetPath "frontend/.env.local"

if ($OpenAIApiKey -and (Test-Path "backend/.env")) {
    $content = Get-Content "backend/.env" -Raw
    if ($content -match "(?m)^OPENAI_API_KEY=") {
        $content = [regex]::Replace($content, "(?m)^OPENAI_API_KEY=.*$", "OPENAI_API_KEY=$OpenAIApiKey")
    } else {
        $content = $content.TrimEnd() + "`r`nOPENAI_API_KEY=$OpenAIApiKey`r`n"
    }
    Set-Content "backend/.env" $content
    Write-Host "Updated OPENAI_API_KEY in backend/.env" -ForegroundColor Green
}

if ($ReconfigureOnly) {
    Write-Host "`nReconfigure mode completed." -ForegroundColor Green
    Write-Host "If needed, run full setup later: powershell -ExecutionPolicy Bypass -File .\scripts\setup-dev.ps1"
    exit 0
}

Write-Step "Installing backend dependencies"
Push-Location "backend"
if (-not (Test-Path ".venv")) {
    python -m venv .venv
}
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt -r requirements-dev.txt
Pop-Location

Write-Step "Installing frontend dependencies"
Push-Location "frontend"
npm install
Pop-Location

Write-Step "Smoke test"
Push-Location "backend"
& ".\.venv\Scripts\python.exe" -m pytest -q tests/test_smoke.py
Pop-Location

Write-Host "`nSetup completed." -ForegroundColor Green
Write-Host "Backend run:  cd backend; .\.venv\Scripts\Activate.ps1; uvicorn app.main:app --reload --port 8000"
Write-Host "Frontend run: cd frontend; npm run dev"
