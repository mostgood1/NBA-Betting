param(
    [int]$Port = 5050,
    [switch]$SkipInstall,
    [switch]$NoBrowser,
    [switch]$Foreground
)

$ErrorActionPreference = 'Stop'

function Write-Info($msg) { Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Err($msg) { Write-Host "[ERR ] $msg" -ForegroundColor Red }

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"

# Ensure venv
if (-not (Test-Path $VenvPython)) {
    Write-Info "Creating virtual environment (.venv)..."
    try {
        py -3 -m venv .venv
    } catch {
        try {
            python -m venv .venv
        } catch {
            Write-Err "Failed to create virtual environment. Ensure Python 3 is installed."
            exit 1
        }
    }
} else {
    Write-Info "Using existing virtual environment."
}

if (-not (Test-Path $VenvPython)) {
    Write-Err "Virtual environment python not found at $VenvPython"
    exit 1
}

# Install requirements (optional)
if (-not $SkipInstall) {
    Write-Info "Installing dependencies..."
    & $VenvPython -m pip install -U pip | Out-Host
    & $VenvPython -m pip install -r requirements.txt | Out-Host
    # Optional: python-dotenv for .env support (app.py tries to load it if present)
    & $VenvPython -m pip show python-dotenv *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Info "Installing python-dotenv (optional)..."
        & $VenvPython -m pip install python-dotenv -q | Out-Null
    }
}

# Try to free the port
try {
    $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($conns) {
        $pids = $conns | Select-Object -ExpandProperty OwningProcess -Unique
        foreach ($procId in $pids) {
            try {
                Write-Warn "Stopping process listening on port $Port (PID $procId)..."
                Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
            } catch {}
        }
    }
} catch {}

Write-Info "Starting Flask app on http://127.0.0.1:$Port ..."
$env:PORT = "$Port"

if ($Foreground) {
    & $VenvPython app.py
} else {
    $p = Start-Process -FilePath $VenvPython -ArgumentList "app.py" -WorkingDirectory $Root -WindowStyle Minimized -PassThru
    Write-Info "Started python.exe (PID $($p.Id))."
    # Wait for readiness (health endpoint) up to ~10s
    $ready = $false
    for ($i = 0; $i -lt 40; $i++) {
        try {
            $resp = Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:$Port/health" -TimeoutSec 2
            if ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 500) { $ready = $true; break }
        } catch {}
        Start-Sleep -Milliseconds 250
    }
    if ($ready) { Write-Info "Health check passed." } else { Write-Warn "Health check did not confirm within timeout." }
    if (-not $NoBrowser) { Start-Process "http://127.0.0.1:$Port/" }
    Write-Host "`nTo stop: Stop-Process -Id $($p.Id) -Force`n" -ForegroundColor Gray
}
