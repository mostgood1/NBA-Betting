param([string]$TaskName = 'NBA-Betting Daily Update')
$ErrorActionPreference = 'Stop'
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
  Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false | Out-Null
  Write-Host "Unregistered scheduled task '$TaskName'"
} else {
  Write-Host "Task '$TaskName' not found"
}
