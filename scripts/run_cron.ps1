# Runs the composite daily cron against the local Flask server
param(
  [string]$Date = (Get-Date -Format 'yyyy-MM-dd'),
  [switch]$Push
)
$token = $env:CRON_TOKEN
if (-not $token) { Write-Host "CRON_TOKEN not set; requests will fall back to ADMIN_KEY if allowed." }
$headers = @{ }
if ($token) { $headers["Authorization"] = "Bearer $token" }
$pushVal = if ($Push) { '1' } else { '0' }
$uri = "http://127.0.0.1:5000/api/cron/run-all?date=$Date&push=$pushVal"
try {
  $resp = Invoke-WebRequest -UseBasicParsing -Headers $headers -Uri $uri -TimeoutSec 120
  $resp.Content | Write-Output
} catch {
  Write-Error $_
}
