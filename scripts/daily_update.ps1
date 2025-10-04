Param(
  [string]$Date = (Get-Date -Format 'yyyy-MM-dd'),
  [switch]$Quiet,
  [string]$LogDir = "logs",
  # If set, stage/commit/pull --rebase/push repo changes (data/processed etc.)
  [switch]$GitPush,
  # If set, do a 'git pull --rebase' before running to reduce conflicts
  [switch]$GitSyncFirst,
  # Optional: Remote server base URL (updated to the correct Render site)
  [string]$RemoteBaseUrl = "https://nba-betting-5qgf.onrender.com"
)

$ErrorActionPreference = 'Stop'

# Resolve repo root from this script
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -Path $Root

# Python resolution (prefer venv)
$VenvPy = Join-Path $Root '.venv\Scripts\python.exe'
$Python = if (Test-Path $VenvPy) { $VenvPy } else { 'python' }

# Logs
$LogPath = Join-Path $Root $LogDir
if (-not (Test-Path $LogPath)) { New-Item -ItemType Directory -Path $LogPath | Out-Null }
$Stamp = (Get-Date).ToString('yyyyMMdd_HHmmss')
$LogFile = Join-Path $LogPath ("local_daily_update_{0}.log" -f $Stamp)

function Write-Log {
  param([string]$Msg)
  $ts = (Get-Date).ToString('u')
  $line = "[$ts] $Msg"
  $line | Out-File -FilePath $LogFile -Append -Encoding UTF8
  if (-not $Quiet) { Write-Host $line }
}

Write-Log "Starting NBA local daily update for date=$Date"
Write-Log "Python: $Python"

# Optionally sync repo to reduce push conflicts
if ($GitSyncFirst) {
  try {
    Write-Log 'Git: pull --rebase'
    & git pull --rebase 2>&1 | Tee-Object -FilePath $LogFile -Append | Out-Null
  } catch { Write-Log ("Git sync failed: {0}" -f $_.Exception.Message) }
}

# Helper to run a python module and record exit codes
function Invoke-PyMod {
  param([string[]]$plist)
  $cmd = @($Python) + $plist
  Write-Log ("Run: {0}" -f ($cmd -join ' '))
  & $Python @plist 2>&1 | Tee-Object -FilePath $LogFile -Append | Out-Null
  return $LASTEXITCODE
}

# If local Flask app is running, prefer calling the composite cron endpoint (does props+predictions+recon)
$BaseUrl = "http://127.0.0.1:5050"
$UseServer = $false
try {
  $resp = Invoke-WebRequest -UseBasicParsing -Uri ($BaseUrl + '/health') -TimeoutSec 3
  if ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 500) { $UseServer = $true }
} catch { $UseServer = $false }

if ($UseServer) {
  Write-Log "Local server detected at $BaseUrl; invoking /api/cron/run-all"
  try {
    $token = $env:CRON_TOKEN
    $headers = @{}
    if ($token) { $headers['Authorization'] = "Bearer $token" }
    $uri = "$BaseUrl/api/cron/run-all?date=$Date&push=0"
    $r = Invoke-WebRequest -UseBasicParsing -Headers $headers -Uri $uri -TimeoutSec 180
    ($r.Content) | Tee-Object -FilePath $LogFile -Append | Out-Null
    Write-Log "run-all completed with status $($r.StatusCode)"
  } catch {
    Write-Log ("run-all failed; falling back to CLI: {0}" -f $_.Exception.Message)
    $UseServer = $false
  }
}

if (-not $UseServer) {
  Write-Log 'Running pipeline via CLI (no server or failed request)'
  # 1) Predictions for the target date (writes data/processed/predictions_<date>.csv and attempts to save odds)
  $rc1 = Invoke-PyMod -plist @('-m','nba_betting.cli','predict-date','--date', $Date)
  Write-Log ("predict-date exit code: {0}" -f $rc1)

  # 2) Reconcile yesterday's games (best-effort; requires server endpoint for meta but CLI not needed)
  try {
    $yesterday = (Get-Date ([datetime]::ParseExact($Date, 'yyyy-MM-dd', $null))).AddDays(-1).ToString('yyyy-MM-dd')
  } catch { $yesterday = (Get-Date).AddDays(-1).ToString('yyyy-MM-dd') }
  Write-Log ("Reconcile games for {0} via server endpoint (if available)" -f $yesterday)
  try {
    $token = $env:CRON_TOKEN
    $headers = @{}
    if ($token) { $headers['Authorization'] = "Bearer $token" }
    $uri = "$BaseUrl/api/cron/reconcile-games?date=$yesterday"
    $r2 = Invoke-WebRequest -UseBasicParsing -Headers $headers -Uri $uri -TimeoutSec 120
    ($r2.Content) | Tee-Object -FilePath $LogFile -Append | Out-Null
  } catch { Write-Log ("reconcile-games call failed: {0}" -f $_.Exception.Message) }

  # 3) Props actuals upsert for yesterday (CLI)
  $rc3 = Invoke-PyMod -plist @('-m','nba_betting.cli','props-actuals','--date', $yesterday)
  Write-Log ("props-actuals exit code: {0}" -f $rc3)

  # 4) Props edges for today (auto source: OddsAPI if available else Bovada)
  $rc4 = Invoke-PyMod -plist @('-m','nba_betting.cli','props-edges','--date', $Date, '--source','auto')
  Write-Log ("props-edges exit code: {0}" -f $rc4)
}

# Simple retention: keep last 21 local_daily_update_* logs
Get-ChildItem -Path $LogPath -Filter 'local_daily_update_*.log' | Sort-Object LastWriteTime -Descending | Select-Object -Skip 21 | ForEach-Object { Remove-Item $_.FullName -ErrorAction SilentlyContinue }

# Optionally commit and push updated artifacts
if ($GitPush) {
  try {
    Write-Log 'Git: staging and pushing updated artifacts'
    & git add -- data data\processed 2>&1 | Tee-Object -FilePath $LogFile -Append | Out-Null
    # Try to include predictions.csv at root if present (legacy)
    if (Test-Path 'predictions.csv') { git add -- predictions.csv | Out-Null }
    $cached = & git diff --cached --name-only
    if ($cached) {
      $msg = "local daily: $Date (predictions/odds/props)"
      & git commit -m $msg 2>&1 | Tee-Object -FilePath $LogFile -Append | Out-Null
      & git pull --rebase 2>&1 | Tee-Object -FilePath $LogFile -Append | Out-Null
      & git push 2>&1 | Tee-Object -FilePath $LogFile -Append | Out-Null
      Write-Log 'Git push complete'
    } else {
      Write-Log 'Git: no staged changes; skipping push'
    }
  } catch {
    Write-Log ("Git push failed: {0}" -f $_.Exception.Message)
  }
}

Write-Log 'Local daily update complete.'
