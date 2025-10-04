param(
  [string]$TaskName = 'NBA-Betting Daily Update',
  [string]$Time = '10:00',   # local time HH:mm
  [switch]$GitPush
)

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$scriptPath = Join-Path $root 'daily_update.ps1'
if (-not (Test-Path $scriptPath)) { throw "daily_update.ps1 not found at $scriptPath" }

# Build the action to invoke PowerShell with our script
$psExe = (Get-Command powershell.exe).Source
$psArgs = @('-NoProfile','-ExecutionPolicy','Bypass','-File',"`"$scriptPath`"")
if ($GitPush) { $psArgs += '-GitPush' }
$argLine = $psArgs -join ' '

$action = New-ScheduledTaskAction -Execute $psExe -Argument $argLine -WorkingDirectory $root

# Daily trigger at specified time
try {
  $hh,$mm = $Time.Split(':')
  $hour = [int]$hh; $min = [int]$mm
} catch { throw "Invalid -Time '$Time' (expected HH:mm)" }
$trigger = New-ScheduledTaskTrigger -Daily -At ([datetime]::Today.AddHours($hour).AddMinutes($min))

# Use current user; task runs whether user is logged on or not is more complex; keep simple
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType InteractiveToken

$task = New-ScheduledTask -Action $action -Trigger $trigger -Principal $principal -Settings (New-ScheduledTaskSettingsSet -StartWhenAvailable)

# Register or update
try {
  if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false | Out-Null
  }
} catch {}

Register-ScheduledTask -TaskName $TaskName -InputObject $task | Out-Null
Write-Host "Registered scheduled task '$TaskName' at $Time (GitPush=$($GitPush.IsPresent))"
