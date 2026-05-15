Param(
  [string]$Date = '',
  [switch]$Quiet,
  [string]$LogDir = "logs",
  # If set, stage/commit/pull --rebase/push repo changes (data/processed etc.)
  [switch]$GitPush,
  # If set, sync from origin/main before running.
  # Clean main branches fast-forward; dirty/diverged local repos fall back to
  # restoring managed processed artifacts for yesterday/today/look-ahead dates.
  [switch]$GitSyncFirst,
  # If set, skip applying totals calibration to predictions (safety valve)
  [switch]$SkipTotalsCalib,
  # Optional: Override the curated slate JSON build query/output (top-N artifact)
  # These map to env vars DAILY_SLATE_QUERY / DAILY_SLATE_QUERY_EXTRA / DAILY_SLATE_OUT
  [string]$SlateQuery,
  [string]$SlateQueryExtra,
  [string]$SlateOut,
  # If set, allow a custom slate query to write to the default output filename
  # (data/processed/recommendations_slate_{Date}.json) instead of *_custom.json.
  [switch]$SlateForceDefaultOut,
  # Optional: Remote server base URL (updated to the correct Render site)
  [string]$RemoteBaseUrl = "https://nba-betting-5qgf.onrender.com",
  # Optional: Bare -CronToken flag is accepted (no value) to avoid task failures
  [switch]$CronToken,
  # Explicit cron token text (overrides env/.env/file discovery)
  [string]$CronTokenParam,
  # Bare -Token should behave like -CronToken (switch); -TokenValue supplies a token string
  [switch]$Token,
  [string]$TokenValue,
  # Build future-safe look-ahead artifacts for Render/UI consumption.
  # -1 = resolve from env DAILY_LOOKAHEAD_DAYS (default 1), 0 = disable.
  [int]$LookAheadDays = -1
)

$ErrorActionPreference = 'Stop'

function Resolve-NbaSlateDate {
  param(
    [string]$TimeZoneId = $env:APP_TZ,
    [int]$CutoffHour = 6
  )

  if ([string]::IsNullOrWhiteSpace($TimeZoneId)) { $TimeZoneId = 'America/New_York' }

  $tzCandidates = @($TimeZoneId)
  switch ($TimeZoneId) {
    'America/New_York' { $tzCandidates += 'Eastern Standard Time' }
    'US/Eastern' { $tzCandidates += 'Eastern Standard Time' }
    'America/Chicago' { $tzCandidates += 'Central Standard Time' }
    'US/Central' { $tzCandidates += 'Central Standard Time' }
    'America/Denver' { $tzCandidates += 'Mountain Standard Time' }
    'US/Mountain' { $tzCandidates += 'Mountain Standard Time' }
    'America/Los_Angeles' { $tzCandidates += 'Pacific Standard Time' }
    'US/Pacific' { $tzCandidates += 'Pacific Standard Time' }
  }

  $tzInfo = $null
  foreach ($tzId in ($tzCandidates | Select-Object -Unique)) {
    if ([string]::IsNullOrWhiteSpace($tzId)) { continue }
    try {
      $tzInfo = [System.TimeZoneInfo]::FindSystemTimeZoneById($tzId)
      if ($null -ne $tzInfo) { break }
    } catch { }
  }

  if ($null -ne $tzInfo) {
    $nowLocal = [System.TimeZoneInfo]::ConvertTimeFromUtc([DateTime]::UtcNow, $tzInfo)
  } else {
    $offsetHours = -5
    if (-not [string]::IsNullOrWhiteSpace($env:APP_TZ_OFFSET_HOURS)) {
      try { $offsetHours = [int]$env:APP_TZ_OFFSET_HOURS } catch { $offsetHours = -5 }
    }
    $nowLocal = [DateTime]::UtcNow.AddHours($offsetHours)
  }

  if ($nowLocal.Hour -lt $CutoffHour) {
    $nowLocal = $nowLocal.AddDays(-1)
  }

  return $nowLocal.ToString('yyyy-MM-dd')
}

function Resolve-NbaSeasonString {
  param([string]$DateValue)

  $dt = [datetime]::ParseExact($DateValue, 'yyyy-MM-dd', $null)
  $seasonYear = if ($dt.Month -ge 7) { $dt.Year } else { $dt.Year - 1 }
  return "{0}-{1}" -f $seasonYear, ("{0:d2}" -f (($seasonYear + 1) % 100))
}

function Resolve-NbaSeasonYear {
  param([string]$DateValue)

  $dt = [datetime]::ParseExact($DateValue, 'yyyy-MM-dd', $null)
  if ($dt.Month -ge 7) {
    return ($dt.Year + 1)
  }
  return $dt.Year
}

function Resolve-RegularSeasonEndDate {
  param(
    [string]$SeasonValue,
    [string]$RepoRootPath
  )

  if ([string]::IsNullOrWhiteSpace($SeasonValue) -or [string]::IsNullOrWhiteSpace($RepoRootPath)) {
    return $null
  }

  $seasonBits = $SeasonValue.Split('-', 2)
  if ($seasonBits.Length -lt 1 -or [string]::IsNullOrWhiteSpace($seasonBits[0])) {
    return $null
  }

  $startYear = $seasonBits[0].Trim()
  if ($startYear.Length -lt 2) {
    return $null
  }

  $seasonTag = $SeasonValue.Replace('-', '_')
  $candidates = @(
    (Join-Path $RepoRootPath ("data/processed/schedule_{0}.csv" -f $seasonTag)),
    (Join-Path $RepoRootPath ("data/raw/schedule_{0}.csv" -f $seasonTag))
  )

  $schedulePath = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
  if ([string]::IsNullOrWhiteSpace($schedulePath)) {
    return $null
  }

  try {
    $seasonSuffix = $startYear.Substring($startYear.Length - 2)
    $regularPrefixes = @(
      ("002{0}" -f $seasonSuffix),
      ("2{0}" -f $seasonSuffix)
    )
    $rows = Import-Csv -Path $schedulePath
    $dates = @(
      $rows |
        Where-Object {
          $gid = [string]$_.game_id
          if ([string]::IsNullOrWhiteSpace($gid)) { return $false }
          $digits = ($gid -replace '[^0-9]', '')
          if ([string]::IsNullOrWhiteSpace($digits)) { return $false }
          $matchedPrefix = $regularPrefixes | Where-Object { $digits.StartsWith($_) } | Select-Object -First 1
          return [bool]$matchedPrefix
        } |
        ForEach-Object {
          try {
            [datetime]::ParseExact([string]$_.date_est, 'yyyy-MM-dd', $null)
          } catch {
            $null
          }
        } |
        Where-Object { $null -ne $_ }
    )
    if ($dates.Count -le 0) {
      return $null
    }
    return (($dates | Sort-Object | Select-Object -Last 1).ToString('yyyy-MM-dd'))
  } catch {
    return $null
  }
}

function Get-PlayoffRetuneDecision {
  param(
    [string]$DateValue,
    [string]$SeasonValue,
    [string]$RepoRootPath
  )

  $decision = [ordered]@{
    Run = $false
    Reason = ''
    RegularSeasonEnd = $null
    ExistingSummary = $null
  }

  try {
    if ($null -ne $env:DAILY_SKIP_PLAYOFF_RETUNE -and $env:DAILY_SKIP_PLAYOFF_RETUNE -match '^(1|true|yes)$') {
      $decision.Reason = 'Skipping playoff retune (DAILY_SKIP_PLAYOFF_RETUNE=1)'
      return $decision
    }

    if ([string]::IsNullOrWhiteSpace($DateValue) -or [string]::IsNullOrWhiteSpace($SeasonValue)) {
      $decision.Reason = 'Skipping playoff retune (date/season unavailable)'
      return $decision
    }

    $regularSeasonEnd = Resolve-RegularSeasonEndDate -SeasonValue $SeasonValue -RepoRootPath $RepoRootPath
    $decision.RegularSeasonEnd = $regularSeasonEnd
    if ([string]::IsNullOrWhiteSpace($regularSeasonEnd)) {
      $decision.Reason = ("Skipping playoff retune (could not resolve regular-season end for {0})" -f $SeasonValue)
      return $decision
    }

    $targetDate = [datetime]::ParseExact($DateValue, 'yyyy-MM-dd', $null)
    $regularSeasonEndDate = [datetime]::ParseExact($regularSeasonEnd, 'yyyy-MM-dd', $null)
    $forceRetune = ($null -ne $env:DAILY_FORCE_PLAYOFF_RETUNE -and $env:DAILY_FORCE_PLAYOFF_RETUNE -match '^(1|true|yes)$')

    if (-not $forceRetune -and $targetDate.Date -le $regularSeasonEndDate.Date) {
      $decision.Reason = ("Skipping playoff retune (date {0} is still within regular season ending {1})" -f $DateValue, $regularSeasonEnd)
      return $decision
    }

    $existing = Get-ChildItem -Path (Join-Path $RepoRootPath ("data/processed/playoff_transition_{0}_*.json" -f $SeasonValue)) -ErrorAction SilentlyContinue |
      Sort-Object -Property LastWriteTimeUtc -Descending |
      Select-Object -First 1
    if ($null -ne $existing) {
      $decision.ExistingSummary = $existing.FullName
    }

    if (-not $forceRetune -and $null -ne $existing) {
      $decision.Reason = ("Skipping playoff retune (existing season summary found: {0})" -f $existing.FullName)
      return $decision
    }

    $decision.Run = $true
    if ($forceRetune) {
      $decision.Reason = ("Running playoff retune (forced) for {0}; regular season ended {1}" -f $SeasonValue, $regularSeasonEnd)
    } else {
      $decision.Reason = ("Running playoff retune for {0}; regular season ended {1}" -f $SeasonValue, $regularSeasonEnd)
    }
    return $decision
  } catch {
    $decision.Reason = ("Skipping playoff retune (decision failed: {0})" -f $_.Exception.Message)
    return $decision
  }
}

function Import-CsvSafe {
  param([string]$Path)

  if ([string]::IsNullOrWhiteSpace($Path) -or -not (Test-Path $Path)) {
    return @()
  }

  try {
    return @(Import-Csv -Path $Path)
  } catch {
    return @()
  }
}

function Format-GroupedCsvCounts {
  param(
    [object[]]$Rows,
    [string]$ColumnName,
    [int]$MaxItems = 8
  )

  if ($null -eq $Rows -or $Rows.Count -le 0 -or [string]::IsNullOrWhiteSpace($ColumnName)) {
    return ''
  }

  $parts = @()
  $sortedGroups = $Rows |
    Group-Object -Property $ColumnName |
    Sort-Object -Property @(
      @{ Expression = 'Count'; Descending = $true },
      @{ Expression = 'Name'; Descending = $false }
    ) |
    Select-Object -First $MaxItems
  foreach ($group in $sortedGroups) {
    $name = [string]$group.Name
    if ([string]::IsNullOrWhiteSpace($name)) { $name = 'UNKNOWN' }
    $parts += ('{0}={1}' -f $name, $group.Count)
  }

  return ($parts -join ', ')
}

function Write-PlayablePropAuditSummary {
  param(
    [string]$AuditKind,
    [string]$RequestedDate,
    [string]$AuditPath,
    [string]$GroupColumn,
    [string]$RollupPath
  )

  $auditRows = Import-CsvSafe -Path $AuditPath
  if ($auditRows.Count -le 0) {
    Write-Log ("Playable prop {0} summary unavailable for {1}; audit CSV missing or empty: {2}" -f $AuditKind, $RequestedDate, $AuditPath)
    return
  }

  $groupSummary = Format-GroupedCsvCounts -Rows $auditRows -ColumnName $GroupColumn
  if ([string]::IsNullOrWhiteSpace($groupSummary)) {
    Write-Log ("Playable prop {0} summary through {1}: rows={2}" -f $AuditKind, $RequestedDate, $auditRows.Count)
  } else {
    Write-Log ("Playable prop {0} summary through {1}: rows={2}; {3}" -f $AuditKind, $RequestedDate, $auditRows.Count, $groupSummary)
  }

  $rollupRows = Import-CsvSafe -Path $RollupPath
  if ($rollupRows.Count -le 0) {
    return
  }

  $top = $rollupRows[0]
  if ($AuditKind -eq 'provider') {
    $playerName = [string]$top.player_name
    $teamAbbr = [string]$top.team_abbr
    $candidateName = [string]$top.top_candidate_name
    $candidateTeam = [string]$top.top_candidate_team_abbr
    $rows = [string]$top.rows
    if (-not [string]::IsNullOrWhiteSpace($playerName) -and -not [string]::IsNullOrWhiteSpace($candidateName)) {
      Write-Log ("Playable prop provider top pair: {0} ({1}) -> {2} ({3}), rows={4}" -f $playerName, $teamAbbr, $candidateName, $candidateTeam, $rows)
    }
    return
  }

  if ($AuditKind -eq 'coverage') {
    $teamAbbr = [string]$top.team_abbr
    $coverageSubtype = [string]$top.coverage_subtype
    $rows = [string]$top.rows
    $players = [string]$top.players
    $dates = [string]$top.dates
    if (-not [string]::IsNullOrWhiteSpace($teamAbbr)) {
      Write-Log ("Playable prop coverage top team: {0} {1}, rows={2}, players={3}, dates={4}" -f $teamAbbr, $coverageSubtype, $rows, $players, $dates)
    }
  }
}

$DateWasImplicit = [string]::IsNullOrWhiteSpace($Date)

# Ignore any token-related parameters to enforce local-only execution
$CronToken = $false
$CronTokenParam = $null
$Token = $false
$TokenValue = $null

# Default behavior: push to Git at the end unless explicitly disabled.
# If caller omitted -GitPush, honor env DAILY_UPDATE_ALWAYS_PUSH (default = true)
if (-not $PSBoundParameters.ContainsKey('GitPush')) {
  $always = $env:DAILY_UPDATE_ALWAYS_PUSH
  if ($null -eq $always -or $always -eq '') { $always = '1' }
  if ($always -match '^(1|true|yes)$') { $GitPush = $true } else { $GitPush = $false }
}

# Default behavior for totals calibration: allow skipping via env DAILY_SKIP_TOTALS_CALIB
if (-not $PSBoundParameters.ContainsKey('SkipTotalsCalib')) {
  $stc = $env:DAILY_SKIP_TOTALS_CALIB
  if ($null -ne $stc -and $stc -match '^(1|true|yes)$') { $SkipTotalsCalib = $true } else { $SkipTotalsCalib = $false }
}

# Default behavior for future-safe look-ahead artifact generation.
# This powers tomorrow's Render/UI slate when today's date has no prebuilt artifacts yet.
if (-not $PSBoundParameters.ContainsKey('LookAheadDays') -or $LookAheadDays -lt 0) {
  $lad = $env:DAILY_LOOKAHEAD_DAYS
  if ($null -eq $lad -or $lad -eq '') { $lad = '1' }
  try { $LookAheadDays = [int]$lad } catch { $LookAheadDays = 1 }
}
if ($LookAheadDays -lt 0) { $LookAheadDays = 0 }
if ($LookAheadDays -gt 3) { $LookAheadDays = 3 }

# Resolve paths
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
# Repo root is the parent of the scripts folder
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir '..')).Path
Set-Location -Path $RepoRoot

# Optional: wire slate override params into env vars (so the existing slate builder block picks them up)
if ($null -ne $SlateQuery -and $SlateQuery -ne '') { $env:DAILY_SLATE_QUERY = $SlateQuery }
if ($null -ne $SlateQueryExtra -and $SlateQueryExtra -ne '') { $env:DAILY_SLATE_QUERY_EXTRA = $SlateQueryExtra }
if ($null -ne $SlateOut -and $SlateOut -ne '') { $env:DAILY_SLATE_OUT = $SlateOut }
if ($SlateForceDefaultOut) { $env:DAILY_SLATE_FORCE_DEFAULT_OUT = '1' }

# Python resolution (prefer local venv which has all dependencies)
$VenvPy = Join-Path $RepoRoot '.venv\Scripts\python.exe'
$NpuPy = 'C:\Users\mostg\OneDrive\Coding\NBA NPU\.venv-arm64\Scripts\python.exe'

# Remove stale git index.lock to avoid interactive prompts during unattended runs
function Remove-StaleGitLock {
  try {
    $lock = Join-Path $RepoRoot '.git/index.lock'
    if (Test-Path $lock) {
      $age = (Get-Date) - (Get-Item $lock).LastWriteTime
      if ($age.TotalSeconds -ge 60) {
        Remove-Item $lock -Force -ErrorAction SilentlyContinue
        Write-Log 'Git: removed stale index.lock'
      } else {
        Write-Log 'Git: index.lock present (fresh); leaving in place'
      }
    }
  } catch {
    Write-Log ("Git: lock cleanup failed: {0}" -f $_.Exception.Message)
  }
}

# First try: use local venv if it exists and has pandas
$Python = $null
if (Test-Path $VenvPy) {
  try {
    & $VenvPy -c "import pandas" 2>$null
    if ($LASTEXITCODE -eq 0) {
      $Python = $VenvPy
      $env:PYTHONPATH = Join-Path $RepoRoot 'src'
      Write-Host "Using local venv with pandas"
    }
  } catch { }
}

# Second try: use NPU environment if local venv failed
if (-not $Python -and (Test-Path $NpuPy)) {
  try {
    & $NpuPy -c "import pandas" 2>$null
    if ($LASTEXITCODE -eq 0) {
      $Python = $NpuPy
      $env:PYTHONPATH = Join-Path $RepoRoot 'src'
      Write-Host "Using NPU venv"
    }
  } catch { }
}

# Fallback to system python
if (-not $Python) {
  $Python = 'python'
  Write-Host "Using system python"
}

# Logs (under repo root)
$LogPath = Join-Path $RepoRoot $LogDir
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

function Invoke-GitCapture {
  param(
    [string[]]$ArgList,
    [switch]$SuppressOutputLog
  )

  $prevEap = $ErrorActionPreference
  $hasNativePref = $false
  $prevNativePref = $null
  try {
    if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
      $hasNativePref = $true
      $prevNativePref = $PSNativeCommandUseErrorActionPreference
      $PSNativeCommandUseErrorActionPreference = $false
    }
    $ErrorActionPreference = 'Continue'
    $output = & git @ArgList 2>&1
    $rc = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $prevEap
    if ($hasNativePref) {
      $PSNativeCommandUseErrorActionPreference = $prevNativePref
    }
  }

  if (-not $SuppressOutputLog -and $null -ne $output) {
    ($output | Out-String).TrimEnd() | Out-File -FilePath $LogFile -Append -Encoding UTF8
  }

  return [pscustomobject]@{ Output = $output; ExitCode = $rc }
}

function Test-ManagedProcessedArtifactName {
  param([string]$Name)

  $allowedPrefixes = @(
    'predictions_',
    'recommendations_',
    'cards_props_snapshot_',
    'cards_sim_detail_',
    'recon_games_',
    'recon_quarters_',
    'recon_props_',
    'recon_players_',
    'boxscores_',
    'boxscore_',
    'pbp_reconcile_',
    'tip_winner_probs_',
    'first_basket_probs_',
    'first_basket_recs_',
    'early_threes_',
    'pbp_metrics_daily_',
    'finals_',
    'closing_lines_',
    'market_',
    'odds_',
    'game_odds_',
    'period_lines_',
    'game_cards_',
    'games_predictions_npu_',
    'oddsapi_player_props_',
    'props_edges_',
    'props_predictions_',
    'props_recommendations_',
    'live_player_lens_tuning_',
    'smart_sim_',
    'daily_artifacts_',
    'injuries_counts_',
    'league_status_'
  )

  foreach ($prefix in $allowedPrefixes) {
    if ($Name.StartsWith($prefix)) {
      return $true
    }
  }
  return $false
}

function Get-DailyManagedSyncDates {
  param(
    [string]$BaseDate,
    [int]$FutureDays = 0
  )

  try {
    $baseDt = [datetime]::ParseExact($BaseDate, 'yyyy-MM-dd', $null)
  } catch {
    return @()
  }

  $dates = @($baseDt.AddDays(-1).ToString('yyyy-MM-dd'), $baseDt.ToString('yyyy-MM-dd'))
  for ($i = 1; $i -le $FutureDays; $i++) {
    $dates += $baseDt.AddDays($i).ToString('yyyy-MM-dd')
  }
  return @($dates | Sort-Object -Unique)
}

function Get-DateScopedManagedLocalArtifacts {
  param([string]$TargetDate)

  $processedDir = Join-Path $RepoRoot 'data\processed'
  if (-not (Test-Path $processedDir)) {
    return @()
  }

  $patterns = @(
    "*_$TargetDate.csv",
    "*_$TargetDate.json",
    "smart_sim_${TargetDate}_*.json",
    "daily_artifacts_${TargetDate}.json"
  )

  $files = @()
  foreach ($pattern in $patterns) {
    $files += Get-ChildItem -Path $processedDir -Filter $pattern -File -ErrorAction SilentlyContinue
  }

  if ($files -and $files.Count -gt 0) {
    $files = $files | Sort-Object FullName -Unique | Where-Object {
      Test-ManagedProcessedArtifactName -Name $_.Name
    }
  }

  return @($files)
}

function Get-DateScopedManagedRemoteArtifacts {
  param(
    [string]$TargetDate,
    [string]$RemoteRef
  )

  $pathspecs = @(
    "data/processed/*_$TargetDate.csv",
    "data/processed/*_$TargetDate.json",
    "data/processed/smart_sim_${TargetDate}_*.json",
    "data/processed/daily_artifacts_${TargetDate}.json"
  )

  $ls = Invoke-GitCapture -ArgList (@('ls-tree', '-r', '--name-only', $RemoteRef, '--') + $pathspecs)
  if ($ls.ExitCode -ne 0) {
    Write-Log ("Git sync: unable to list remote managed artifacts for {0} from {1} (exit={2})" -f $TargetDate, $RemoteRef, $ls.ExitCode)
    return @()
  }

  $paths = @()
  foreach ($item in @($ls.Output)) {
    $path = [string]$item
    if ([string]::IsNullOrWhiteSpace($path)) { continue }
    $trimmed = $path.Trim()
    $name = [System.IO.Path]::GetFileName($trimmed)
    if (Test-ManagedProcessedArtifactName -Name $name) {
      $paths += $trimmed
    }
  }

  return @($paths | Sort-Object -Unique)
}

function Get-DateScopedManagedRenderArtifacts {
  param(
    [string]$TargetDate,
    [string]$BaseUrl
  )

  if ([string]::IsNullOrWhiteSpace($TargetDate) -or [string]::IsNullOrWhiteSpace($BaseUrl)) {
    return [pscustomobject]@{ Success = $false; Items = @() }
  }

  $base = $BaseUrl.TrimEnd('/')
  $patterns = @(
    "*_$TargetDate.csv",
    "*_$TargetDate.json",
    "smart_sim_${TargetDate}_*.json",
    "daily_artifacts_${TargetDate}.json"
  )

  $items = @()
  foreach ($pattern in $patterns) {
    try {
      $uri = "{0}/api/list/processed?pattern={1}" -f $base, [System.Uri]::EscapeDataString($pattern)
      $resp = Invoke-RestMethod -Uri $uri -Method Get -TimeoutSec 30
      foreach ($item in @($resp.items)) {
        $name = [string]$item.name
        if ([string]::IsNullOrWhiteSpace($name)) { continue }
        if (Test-ManagedProcessedArtifactName -Name $name) {
          $items += $name
        }
      }
    } catch {
      Write-Log ("Render sync: failed to list artifacts for {0} with pattern {1}: {2}" -f $TargetDate, $pattern, $_.Exception.Message)
      return [pscustomobject]@{ Success = $false; Items = @() }
    }
  }

  return [pscustomobject]@{ Success = $true; Items = @($items | Sort-Object -Unique) }
}

function Restore-ManagedArtifactsFromRemote {
  param(
    [string[]]$RemotePaths,
    [string]$RemoteRef
  )

  foreach ($remotePath in @($RemotePaths)) {
    if ([string]::IsNullOrWhiteSpace($remotePath)) { continue }
    $show = Invoke-GitCapture -ArgList @('show', ("{0}:{1}" -f $RemoteRef, $remotePath)) -SuppressOutputLog
    if ($show.ExitCode -ne 0) {
      Write-Log ("Git sync: failed to restore {0} from {1} (exit={2})" -f $remotePath, $RemoteRef, $show.ExitCode)
      continue
    }

    $dest = Join-Path $RepoRoot ($remotePath -replace '/', '\\')
    $destDir = Split-Path -Parent $dest
    if (-not (Test-Path $destDir)) {
      New-Item -ItemType Directory -Path $destDir -Force | Out-Null
    }

    $lines = @($show.Output | ForEach-Object { [string]$_ })
    $content = [string]::Join([Environment]::NewLine, $lines)
    if ($lines.Count -gt 0) {
      $content += [Environment]::NewLine
    }
    [System.IO.File]::WriteAllText($dest, $content, [System.Text.UTF8Encoding]::new($false))
  }
}

function Restore-ManagedArtifactsFromRender {
  param(
    [string[]]$Names,
    [string]$BaseUrl
  )

  if ([string]::IsNullOrWhiteSpace($BaseUrl)) {
    return 0
  }

  $base = $BaseUrl.TrimEnd('/')
  $restored = 0
  foreach ($name in @($Names)) {
    if ([string]::IsNullOrWhiteSpace($name)) { continue }

    $dest = Join-Path $RepoRoot (Join-Path 'data\processed' $name)
    $destDir = Split-Path -Parent $dest
    if (-not (Test-Path $destDir)) {
      New-Item -ItemType Directory -Path $destDir -Force | Out-Null
    }

    $tmp = [System.IO.Path]::GetTempFileName()
    try {
      $uri = "{0}/api/processed/download?name={1}" -f $base, [System.Uri]::EscapeDataString($name)
      Invoke-WebRequest -Uri $uri -Method Get -OutFile $tmp -TimeoutSec 60 | Out-Null
      Move-Item -Path $tmp -Destination $dest -Force
      $restored += 1
    } catch {
      if (Test-Path $tmp) {
        Remove-Item -Path $tmp -Force -ErrorAction SilentlyContinue
      }
      Write-Log ("Render sync: failed to download {0}: {1}" -f $name, $_.Exception.Message)
    }
  }

  return $restored
}

function Sync-DateScopedManagedArtifactsFromRemote {
  param(
    [string]$TargetDate,
    [string]$Remote = 'origin',
    [string]$Branch = 'main'
  )

  if ([string]::IsNullOrWhiteSpace($TargetDate)) {
    return
  }

  $remoteRef = "{0}/{1}" -f $Remote, $Branch
  $localFiles = @(Get-DateScopedManagedLocalArtifacts -TargetDate $TargetDate)
  $remotePaths = @(Get-DateScopedManagedRemoteArtifacts -TargetDate $TargetDate -RemoteRef $remoteRef)

  if ($localFiles.Count -eq 0 -and $remotePaths.Count -eq 0) {
    Write-Log ("Git sync: no managed date-scoped artifacts found locally or remotely for {0}" -f $TargetDate)
    return
  }

  foreach ($file in $localFiles) {
    try {
      Remove-Item -Path $file.FullName -Force -ErrorAction Stop
    } catch {
      Write-Log ("Git sync: failed to remove local artifact {0}: {1}" -f $file.FullName, $_.Exception.Message)
    }
  }

  if ($remotePaths.Count -le 0) {
    Write-Log ("Git sync: cleared local managed artifacts for {0}; none exist on {1}" -f $TargetDate, $remoteRef)
    return
  }

  Restore-ManagedArtifactsFromRemote -RemotePaths $remotePaths -RemoteRef $remoteRef
  Write-Log ("Git sync: restored {0} managed artifacts for {1} from {2}" -f $remotePaths.Count, $TargetDate, $remoteRef)
}

function Sync-DateScopedManagedArtifactsFromRender {
  param(
    [string]$TargetDate,
    [string]$BaseUrl
  )

  if ([string]::IsNullOrWhiteSpace($TargetDate) -or [string]::IsNullOrWhiteSpace($BaseUrl)) {
    return [pscustomobject]@{ Success = $false; RemoteCount = 0; RestoredCount = 0 }
  }

  $localFiles = @(Get-DateScopedManagedLocalArtifacts -TargetDate $TargetDate)
  $renderList = Get-DateScopedManagedRenderArtifacts -TargetDate $TargetDate -BaseUrl $BaseUrl
  if (-not $renderList.Success) {
    return [pscustomobject]@{ Success = $false; RemoteCount = 0; RestoredCount = 0 }
  }

  $remoteNames = @($renderList.Items)
  if ($localFiles.Count -eq 0 -and $remoteNames.Count -eq 0) {
    Write-Log ("Render sync: no managed date-scoped artifacts found locally or on Render for {0}" -f $TargetDate)
    return [pscustomobject]@{ Success = $true; RemoteCount = 0; RestoredCount = 0 }
  }

  foreach ($file in $localFiles) {
    try {
      Remove-Item -Path $file.FullName -Force -ErrorAction Stop
    } catch {
      Write-Log ("Render sync: failed to remove local artifact {0}: {1}" -f $file.FullName, $_.Exception.Message)
    }
  }

  if ($remoteNames.Count -le 0) {
    Write-Log ("Render sync: cleared local managed artifacts for {0}; none exist on Render" -f $TargetDate)
    return [pscustomobject]@{ Success = $true; RemoteCount = 0; RestoredCount = 0 }
  }

  $restored = Restore-ManagedArtifactsFromRender -Names $remoteNames -BaseUrl $BaseUrl
  Write-Log ("Render sync: restored {0}/{1} managed artifacts for {2} from {3}" -f $restored, $remoteNames.Count, $TargetDate, $BaseUrl)
  return [pscustomobject]@{ Success = $true; RemoteCount = $remoteNames.Count; RestoredCount = $restored }
}

# Load .env (if present) into the current PowerShell process environment so child Python sees keys (e.g., ODDS_API_KEY)
function Import-DotEnv {
  param([string]$Path)
  if (-not (Test-Path $Path)) { return }
  try {
    Get-Content -Path $Path -Encoding UTF8 | ForEach-Object {
      $line = $_.Trim()
      if (-not $line) { return }
      if ($line.StartsWith('#')) { return }
      $idx = $line.IndexOf('=')
      if ($idx -lt 1) { return }
      $key = $line.Substring(0, $idx).Trim()
      $val = $line.Substring($idx + 1).Trim().Trim('"').Trim("'")
  if ($key) { Set-Item -Path "Env:$key" -Value $val }
    }
    Write-Log "Loaded environment from .env"
  } catch {
    Write-Log (".env load failed (non-fatal): {0}" -f $_.Exception.Message)
  }
}

# Import .env at repo root if available
$DotEnvPath = Join-Path $RepoRoot '.env'
Import-DotEnv -Path $DotEnvPath

if ($DateWasImplicit) {
  $Date = Resolve-NbaSlateDate
}

# Ensure Python writes UTF-8 to stdout/stderr to avoid UnicodeEncodeError on Windows PowerShell consoles
$env:PYTHONIOENCODING = 'utf-8'

# Avoid benign warnings on stderr (e.g., pandas DtypeWarning) being treated as errors
# by PowerShell in some environments/tasks.
$env:PYTHONWARNINGS = 'ignore'

# Reduce noisy ONNXRuntime/CPU info warnings (especially on Windows ARM).
# Safe to set even if ignored by the runtime.
$env:ONNXRUNTIME_LOG_SEVERITY_LEVEL = '3'
$env:ORT_DISABLE_CPUINFO = '1'

# PowerShell 7+: avoid treating native stderr as error records.
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
  $PSNativeCommandUseErrorActionPreference = $false
}

Write-Log "Starting NBA local daily update for date=$Date"
if ($DateWasImplicit) {
  Write-Log "Date default resolved from US local slate time (APP_TZ or America/New_York, 6am cutoff)"
}
Write-Log "Python: $Python"

$IsCiRun = $false
try {
  $ga = $env:GITHUB_ACTIONS
  $ci = $env:CI
  if (($null -ne $ga -and $ga -match '^(1|true|yes)$') -or ($null -ne $ci -and $ci -match '^(1|true|yes)$')) {
    $IsCiRun = $true
  }
} catch {
  $IsCiRun = $false
}

if ($IsCiRun) {
  if ($null -eq $env:DAILY_SKIP_HISTORICAL_MAINTENANCE -or $env:DAILY_SKIP_HISTORICAL_MAINTENANCE -eq '') {
    $env:DAILY_SKIP_HISTORICAL_MAINTENANCE = '1'
  }
  if ($null -eq $env:DAILY_SMARTSIM_NSIMS -or $env:DAILY_SMARTSIM_NSIMS -eq '') {
    $env:DAILY_SMARTSIM_NSIMS = '500'
  }
  if ($null -eq $env:DAILY_BUILD_LEAGUE_STATUS_TIMEOUT_SEC -or $env:DAILY_BUILD_LEAGUE_STATUS_TIMEOUT_SEC -eq '') {
    $env:DAILY_BUILD_LEAGUE_STATUS_TIMEOUT_SEC = '900'
  }
  if ($null -eq $env:DAILY_REQUIRE_PROPS_LINES -or $env:DAILY_REQUIRE_PROPS_LINES -eq '') {
    $env:DAILY_REQUIRE_PROPS_LINES = '0'
  }
  if ($null -eq $env:DAILY_SKIP_PLAYER_AUDITS -or $env:DAILY_SKIP_PLAYER_AUDITS -eq '') {
    $env:DAILY_SKIP_PLAYER_AUDITS = '1'
  }
  if ($null -eq $env:DAILY_SKIP_ROSTER_AUDIT -or $env:DAILY_SKIP_ROSTER_AUDIT -eq '') {
    $env:DAILY_SKIP_ROSTER_AUDIT = '1'
  }
  if ($null -eq $env:DAILY_SKIP_YESTERDAY_ROSTER_AUDIT -or $env:DAILY_SKIP_YESTERDAY_ROSTER_AUDIT -eq '') {
    $env:DAILY_SKIP_YESTERDAY_ROSTER_AUDIT = '1'
  }
  Write-Log ("CI runtime profile: skip_historical_maintenance={0}, smartsim_n_sims={1}, league_status_timeout_sec={2}, require_props_lines={3}, skip_player_audits={4}, skip_roster_audit={5}, skip_yesterday_roster_audit={6}" -f $env:DAILY_SKIP_HISTORICAL_MAINTENANCE, $env:DAILY_SMARTSIM_NSIMS, $env:DAILY_BUILD_LEAGUE_STATUS_TIMEOUT_SEC, $env:DAILY_REQUIRE_PROPS_LINES, $env:DAILY_SKIP_PLAYER_AUDITS, $env:DAILY_SKIP_ROSTER_AUDIT, $env:DAILY_SKIP_YESTERDAY_ROSTER_AUDIT)
}

# Schedule gating: on no-game days, continue but anchor reconciliation to the last slate date.
$NoSlateDay = $false
$LastSlateDate = $null
try {
  $jf = Join-Path $RepoRoot 'data\processed\schedule_2025_26.json'
  if (Test-Path $jf) {
    $raw = Get-Content -Path $jf -Raw
    $sched = $raw | ConvertFrom-Json
    $games = 0
    try {
      $games = @($sched | Where-Object {
        try {
          $d = $_.date_utc
          if ($null -eq $d) { return $false }
          $ds = ([datetime]::Parse($d)).ToString('yyyy-MM-dd')
          return $ds -eq $Date
        } catch {
          return ($_.date_utc -eq $Date)
        }
      }).Count
    } catch {
      $games = 0
    }
    if ($games -le 0) {
      $NoSlateDay = $true
      # Find the most recent slate date on or before $Date
      try {
        $allDates = @()
        foreach ($g in @($sched)) {
          try {
            $d = $g.date_utc
            if ($null -eq $d) { continue }
            $ds = ([datetime]::Parse($d)).ToString('yyyy-MM-dd')
            if ($ds) { $allDates += $ds }
          } catch {
            try {
              $ds2 = [string]$g.date_utc
              if ($ds2) {
                $allDates += $ds2
              }
            } catch { }
          }
        }
        $unique = $allDates | Where-Object { $_ -and $_ -le $Date } | Sort-Object -Unique
        if ($unique -and $unique.Count -gt 0) {
          $LastSlateDate = $unique[-1]
        }
      } catch {
        $LastSlateDate = $null
      }

      if ($LastSlateDate) {
        Write-Log ("No NBA games scheduled for {0}; will reconcile using last slate date {1}" -f $Date, $LastSlateDate)
      } else {
        Write-Log ("No NBA games scheduled for {0}; could not find a last slate date (continuing anyway)" -f $Date)
      }
    } else {
      Write-Log ("Slate size: {0} games" -f $games)
    }
  } else {
    Write-Log "Schedule file not found; skipping schedule gating"
  }
} catch { Write-Log ("Schedule gating failed (continuing): {0}" -f $_.Exception.Message) }

# Disable remote server authentication; enforce local-only execution
Write-Log "Server auth: disabled; enforcing local-only run"

# Optionally sync repo to reduce push conflicts
if ($GitSyncFirst) {
  try {
    $branchInfo = Invoke-GitCapture -ArgList @('branch', '--show-current')
    $currentBranch = (($branchInfo.Output | Out-String).Trim())
    if ([string]::IsNullOrWhiteSpace($currentBranch)) { $currentBranch = '(unknown)' }

    $statusInfo = Invoke-GitCapture -ArgList @('status', '--porcelain')
    $hasLocalChanges = -not [string]::IsNullOrWhiteSpace((($statusInfo.Output | Out-String).Trim()))

    Write-Log 'Git sync: fetch origin main'
    $fetchInfo = Invoke-GitCapture -ArgList @('fetch', 'origin', 'main')
    if ($fetchInfo.ExitCode -ne 0) {
      Write-Log ("Git sync failed during fetch (exit={0})" -f $fetchInfo.ExitCode)
    } else {
      $ahead = -1
      $behind = -1
      $countsInfo = Invoke-GitCapture -ArgList @('rev-list', '--left-right', '--count', 'HEAD...origin/main')
      if ($countsInfo.ExitCode -eq 0) {
        $parts = ((($countsInfo.Output | Out-String).Trim()) -split '\s+')
        if ($parts.Count -ge 2) {
          try {
            $ahead = [int]$parts[0]
            $behind = [int]$parts[1]
          } catch {
            $ahead = -1
            $behind = -1
          }
        }
      }

      Write-Log ("Git sync: branch={0} ahead={1} behind={2} dirty={3}" -f $currentBranch, $ahead, $behind, $hasLocalChanges)
      $didFastForwardPull = $false
      if ($currentBranch -eq 'main' -and -not $hasLocalChanges -and $ahead -eq 0 -and $behind -gt 0) {
        Write-Log 'Git sync: pull --ff-only origin main'
        $pullInfo = Invoke-GitCapture -ArgList @('pull', '--ff-only', 'origin', 'main')
        if ($pullInfo.ExitCode -eq 0) {
          $didFastForwardPull = $true
          Write-Log 'Git sync: fast-forward pull succeeded'
        } else {
          Write-Log ("Git sync: fast-forward pull failed (exit={0}); falling back to managed artifact restore" -f $pullInfo.ExitCode)
        }
      } elseif ($currentBranch -eq 'main' -and -not $hasLocalChanges -and $ahead -eq 0 -and $behind -eq 0) {
        Write-Log 'Git sync: local repo already matches origin/main'
      } else {
        Write-Log 'Git sync: skipping full repo pull; using managed artifact restore fallback'
      }

      if (-not $didFastForwardPull) {
        $syncDates = @(Get-DailyManagedSyncDates -BaseDate $Date -FutureDays $LookAheadDays)
        foreach ($syncDate in $syncDates) {
          $renderSync = $null
          if (-not [string]::IsNullOrWhiteSpace($RemoteBaseUrl)) {
            $renderSync = Sync-DateScopedManagedArtifactsFromRender -TargetDate $syncDate -BaseUrl $RemoteBaseUrl
          }
          if ($null -eq $renderSync -or -not $renderSync.Success) {
            Sync-DateScopedManagedArtifactsFromRemote -TargetDate $syncDate -Remote 'origin' -Branch 'main'
          }
        }
      }
    }
  } catch { Write-Log ("Git sync failed: {0}" -f $_.Exception.Message) }
}

# Helper to run a python module and record exit codes
function Write-StreamToLogAndHost {
  param(
    [Parameter(ValueFromPipeline = $true)]
    [AllowNull()]
    [object]$InputObject
  )
  process {
    if ($null -eq $InputObject) { return }
    $InputObject | Tee-Object -FilePath $LogFile -Append | Out-Host
  }
}

function Invoke-PyMod {
  param([string[]]$plist)
  $cmd = @($Python) + $plist
  Write-Log ("Run: {0}" -f ($cmd -join ' '))
  # Capture both stdout and stderr, but don't fail on stderr output
  $ErrorActionPreference = 'Continue'
  $nativePrefSupported = $false
  $nativePrefPrevious = $null
  try {
    $nativePrefVar = Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue
    if ($null -ne $nativePrefVar) {
      $nativePrefSupported = $true
      $nativePrefPrevious = [bool]$nativePrefVar.Value
      Set-Variable -Name PSNativeCommandUseErrorActionPreference -Scope Local -Value $false
    }
    & $Python @plist 2>&1 | Write-StreamToLogAndHost
    $exitCode = $LASTEXITCODE
  } finally {
    if ($nativePrefSupported) {
      Set-Variable -Name PSNativeCommandUseErrorActionPreference -Scope Local -Value $nativePrefPrevious
    }
    $ErrorActionPreference = 'Stop'
  }
  return $exitCode
}

function Invoke-LoggedNativeCommand {
  param(
    [Parameter(Mandatory = $true)]
    [string]$FilePath,
    [string[]]$ArgumentList = @()
  )

  $stdoutPath = [System.IO.Path]::GetTempFileName()
  $stderrPath = [System.IO.Path]::GetTempFileName()
  $exitCode = 0
  try {
    $process = Start-Process -FilePath $FilePath `
      -ArgumentList $ArgumentList `
      -NoNewWindow `
      -Wait `
      -PassThru `
      -RedirectStandardOutput $stdoutPath `
      -RedirectStandardError $stderrPath `
      -ErrorAction Stop

    if (Test-Path $stdoutPath) {
      Get-Content -Path $stdoutPath -ErrorAction SilentlyContinue | Tee-Object -FilePath $LogFile -Append | Out-Host
    }

    if (Test-Path $stderrPath) {
      Get-Content -Path $stderrPath -ErrorAction SilentlyContinue | Tee-Object -FilePath $LogFile -Append | Out-Host
    }

    if ($null -ne $process) {
      $exitCode = [int]$process.ExitCode
    }
  } finally {
    Remove-Item -Path $stdoutPath -ErrorAction SilentlyContinue
    Remove-Item -Path $stderrPath -ErrorAction SilentlyContinue
  }

  if ($null -eq $exitCode) {
    $exitCode = 0
  }

  return [int]$exitCode
}

function Format-ExitCodeForLog {
  param([object]$Value)

  $items = @($Value)
  if ($items.Count -eq 0) {
    return 'unknown'
  }

  $last = $items[-1]
  if ($null -eq $last) {
    return 'unknown'
  }

  $text = [string]$last
  if ([string]::IsNullOrWhiteSpace($text)) {
    return 'unknown'
  }

  return $text.Trim()
}

function Test-CsvHasDataRows {
  param([string]$Path)
  if (-not (Test-Path $Path)) {
    return $false
  }
  try {
    $lines = Get-Content -Path $Path -TotalCount 2 -ErrorAction Stop
    if ($null -eq $lines) {
      return $false
    }
    $arr = @($lines)
    if ($arr.Count -lt 2) {
      return $false
    }
    return (-not [string]::IsNullOrWhiteSpace([string]$arr[1]))
  } catch {
    return $false
  }
}

function Get-CsvDataRowCount {
  param([string]$Path)
  if (-not (Test-Path $Path)) {
    return 0
  }
  try {
    $rows = Import-Csv -Path $Path -ErrorAction Stop
    if ($null -eq $rows) {
      return 0
    }
    return @($rows).Count
  } catch {
    return 0
  }
}

function Get-PropsRecommendationsPlayableRowCount {
  param([string]$Path)
  if (-not (Test-Path $Path)) {
    return 0
  }
  try {
    $rows = Import-Csv -Path $Path -ErrorAction Stop
    if ($null -eq $rows) {
      return 0
    }
    $count = 0
    foreach ($row in @($rows)) {
      $plays = [string]$row.plays
      if ([string]::IsNullOrWhiteSpace($plays)) {
        continue
      }
      $trim = $plays.Trim()
      if ($trim.ToLower() -in @('nan', 'none', 'null')) {
        continue
      }
      if ($trim -match '^\[\s*\]$') {
        continue
      }
      $count += 1
    }
    return $count
  } catch {
    return 0
  }
}

function Get-LeagueStatusSlateTeamCount {
  param([string]$Path)
  if (-not (Test-Path $Path)) {
    return 0
  }
  try {
    $rows = Import-Csv -Path $Path -ErrorAction Stop
    if ($null -eq $rows) {
      return 0
    }
    return @(
      $rows |
        Where-Object { [string]$_.team_on_slate -match '^(1|true|yes)$' } |
        Select-Object -ExpandProperty team -Unique
    ).Count
  } catch {
    return 0
  }
}

function Get-GameOddsSlateTeamCount {
  param([string]$Path)
  if (-not (Test-Path $Path)) {
    return 0
  }
  try {
    $rows = Import-Csv -Path $Path -ErrorAction Stop
    if ($null -eq $rows) {
      return 0
    }
    $teams = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($row in @($rows)) {
      $homeTeam = [string]$row.home_team
      $awayTeam = [string]$row.visitor_team
      if ([string]::IsNullOrWhiteSpace($awayTeam)) {
        $awayTeam = [string]$row.away_team
      }
      if (-not [string]::IsNullOrWhiteSpace($homeTeam)) {
        $null = $teams.Add($homeTeam.Trim())
      }
      if (-not [string]::IsNullOrWhiteSpace($awayTeam)) {
        $null = $teams.Add($awayTeam.Trim())
      }
    }
    return $teams.Count
  } catch {
    return 0
  }
}

function Repair-LeagueStatusFromGameOdds {
  param(
    [string]$TargetDate,
    [int]$TimeoutSeconds
  )

  $leagueStatusPath = Join-Path $RepoRoot ("data/processed/league_status_{0}.csv" -f $TargetDate)
  $gameOddsPath = Join-Path $RepoRoot ("data/processed/game_odds_{0}.csv" -f $TargetDate)

  $gameOddsTeams = Get-GameOddsSlateTeamCount -Path $gameOddsPath
  $leagueStatusTeams = Get-LeagueStatusSlateTeamCount -Path $leagueStatusPath
  Write-Log ("League-status repair check: game_odds_teams={0}, league_status_slate_teams={1}" -f $gameOddsTeams, $leagueStatusTeams)

  if ($gameOddsTeams -le 0 -or $leagueStatusTeams -gt 0) {
    return
  }

  Write-Log ("Rebuilding league_status for {0} because game_odds shows a slate but team_on_slate is empty" -f $TargetDate)
  $rcRepair = Invoke-PyModWithTimeout -plist @('-m','nba_betting.cli','build-league-status','--date', $TargetDate) -TimeoutSeconds $TimeoutSeconds -Label 'build_league_status_post_odds'
  Write-Log ("build-league-status post-odds exit code: {0}" -f $rcRepair)

  $leagueStatusTeamsAfter = Get-LeagueStatusSlateTeamCount -Path $leagueStatusPath
  Write-Log ("League-status repair result: league_status_slate_teams={0}" -f $leagueStatusTeamsAfter)
  if ($gameOddsTeams -gt 0 -and $leagueStatusTeamsAfter -le 0) {
    throw ("league_status still has no slate teams after post-odds rebuild for {0}" -f $TargetDate)
  }
}

function Invoke-SharedPropsRefreshWorker {
  param([string]$TargetDate)

  $bookmakers = [string]$env:PLAYER_PROP_BOOKMAKERS
  $bookmakersDisplay = if ([string]::IsNullOrWhiteSpace($bookmakers)) { 'all-us-region-books' } else { $bookmakers }

  $workerLog = Join-Path $LogPath ("shared_props_refresh_{0}_{1}.log" -f $TargetDate, $Stamp)
  $payload = [ordered]@{
    date_str   = $TargetDate
    regions    = 'us'
    bookmakers = $bookmakers
    markets    = ''
    do_edges   = $true
    do_export  = $true
    do_push    = $false
    log_file   = $workerLog
    started_at = [DateTime]::UtcNow.ToString('o')
  } | ConvertTo-Json -Compress -Depth 5

  $env:NBA_BETTING_ODDSAPI_PROPS_JOB = $payload
  try {
    Write-Log ("Running shared OddsAPI props refresh worker for {0} (bookmakers={1})" -f $TargetDate, $bookmakersDisplay)
    $rc = Invoke-PyMod -plist @('-m', 'nba_betting.refresh_oddsapi_props_job')
    Write-Log ("refresh_oddsapi_props_job exit code: {0}" -f $rc)
  } finally {
    Remove-Item Env:NBA_BETTING_ODDSAPI_PROPS_JOB -ErrorAction SilentlyContinue
  }

  $rawSnapshotPath = Join-Path $RepoRoot ("data/raw/odds_nba_player_props_{0}.csv" -f $TargetDate)
  $snapshotAliasPath = Join-Path $RepoRoot ("data/processed/oddsapi_player_props_{0}.csv" -f $TargetDate)
  $edgesPath = Join-Path $RepoRoot ("data/processed/props_edges_{0}.csv" -f $TargetDate)
  $propsRecsPath = Join-Path $RepoRoot ("data/processed/props_recommendations_{0}.csv" -f $TargetDate)

  $snapshotRows = Get-CsvDataRowCount -Path $rawSnapshotPath
  $aliasRows = Get-CsvDataRowCount -Path $snapshotAliasPath
  $edgesRows = Get-CsvDataRowCount -Path $edgesPath
  $recsRows = Get-CsvDataRowCount -Path $propsRecsPath

  Write-Log (
    "Shared props refresh summary: snapshot_rows={0}, alias_rows={1}, edges_rows={2}, rec_rows={3}, log={4}" -f
      $snapshotRows, $aliasRows, $edgesRows, $recsRows, $workerLog
  )

  if ($rc -ne 0) {
    return $rc
  }
  if ($snapshotRows -gt 0 -and $aliasRows -le 0) {
    Write-Log 'Shared props refresh wrote snapshot rows but no processed oddsapi_player_props alias.'
    return 1
  }
  if ($snapshotRows -gt 0 -and $edgesRows -le 0) {
    Write-Log 'Shared props refresh wrote snapshot rows but props_edges remained empty.'
    return 1
  }
  if ($snapshotRows -gt 0 -and $recsRows -le 0) {
    Write-Log 'Shared props refresh wrote snapshot rows but props_recommendations remained empty.'
    return 1
  }

  return 0
}

# Helper to run a python module with a hard timeout (prevents hangs on network calls)
function Invoke-PyModWithTimeout {
  param(
    [string[]]$plist,
    [int]$TimeoutSeconds = 180,
    [string]$Label = 'python'
  )
  $cmd = @($Python) + $plist
  Write-Log ("Run (timeout={0}s): {1}" -f $TimeoutSeconds, ($cmd -join ' '))

  $tmpBase = Join-Path $LogPath ("py_{0}_{1}_{2}" -f $Label, $Stamp, ([guid]::NewGuid().ToString('N')))
  $outStd = "$tmpBase.out"
  $outErr = "$tmpBase.err"

  try {
    $p = Start-Process -FilePath $Python -ArgumentList $plist -NoNewWindow -PassThru -RedirectStandardOutput $outStd -RedirectStandardError $outErr
  } catch {
    Write-Log ("Start-Process failed: {0}" -f $_.Exception.Message)
    return 1
  }

  # IMPORTANT: Wait-Process returns *no output* unless -PassThru is used.
  # Using its return value as a boolean causes false timeouts.
  $finished = $false
  try {
    $ms = [int]([Math]::Max(1, $TimeoutSeconds) * 1000)
    $finished = $p.WaitForExit($ms)
  } catch {
    try {
      $p.Refresh()
      $finished = [bool]$p.HasExited
    } catch {
      $finished = $false
    }
  }

  if (-not $finished) {
    # On Windows, a venv python.exe may be a launcher that spawns a child interpreter.
    # Kill the entire process tree so we don't leave orphaned python.exe processes behind.
    $killed = $false
    try {
      & taskkill /PID $p.Id /T /F 2>&1 | Out-Null
      $killed = $true
    } catch {
      $killed = $false
    }
    if (-not $killed) {
      try { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } catch {}
    }
    Write-Log ("TIMEOUT: killed process tree after {0}s (pid={1})" -f $TimeoutSeconds, $p.Id)
    try {
      if (Test-Path $outStd) { Get-Content -Path $outStd -Raw | Write-StreamToLogAndHost }
      if (Test-Path $outErr) { Get-Content -Path $outErr -Raw | Write-StreamToLogAndHost }
    } catch {}
    try { Remove-Item -Force -ErrorAction SilentlyContinue $outStd, $outErr } catch {}
    return 124
  }

  # Ensure process fully exited and output handles are released.
  try { $p.WaitForExit() } catch {}

  try {
    if (Test-Path $outStd) { Get-Content -Path $outStd -Raw | Write-StreamToLogAndHost }
    if (Test-Path $outErr) { Get-Content -Path $outErr -Raw | Write-StreamToLogAndHost }
  } catch {}

  $exitCode = 0
  try { $p.Refresh() } catch {}
  try { $exitCode = $p.ExitCode } catch { $exitCode = 0 }
  if ($null -eq $exitCode) { $exitCode = 0 }
  try { Remove-Item -Force -ErrorAction SilentlyContinue $outStd, $outErr } catch {}
  return [int]$exitCode
}

# Helper: check if a file exists and is "fresh" (recently updated).
function Test-FreshFile {
  param(
    [string]$Path,
    [int]$MaxAgeMinutes = 60
  )
  try {
    if (-not (Test-Path $Path)) { return $false }
    if ($MaxAgeMinutes -le 0) { return $true }
    $age = (Get-Date) - (Get-Item $Path).LastWriteTime
    return ($age.TotalMinutes -le [double]$MaxAgeMinutes)
  } catch {
    return $false
  }
}

function Get-RosterTeamCount {
  param([string]$Path)

  try {
    if (-not (Test-Path $Path)) { return 0 }
    $rows = Import-Csv -Path $Path
    if ($null -eq $rows) { return 0 }
    $teams = @(
      $rows |
        ForEach-Object { [string]($_.TEAM_ABBREVIATION) } |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
        ForEach-Object { $_.Trim().ToUpperInvariant() } |
        Sort-Object -Unique
    )
    return $teams.Count
  } catch {
    return 0
  }
}

# Helper to run the connected realism evaluation (used by the full pipeline and an optional "only" mode)
function Invoke-ConnectedRealismEval {
  $skipConn = $env:DAILY_SKIP_CONNECTED_REALISM
  if ($null -ne $skipConn -and $skipConn -match '^(1|true|yes)$') {
    Write-Log 'Skipping connected realism (DAILY_SKIP_CONNECTED_REALISM=1)'
    return 0
  }

  $connDays = $env:DAILY_CONNECTED_REALISM_DAYS
  if ($null -eq $connDays -or $connDays -notmatch '^\d+$') { $connDays = '1' }
  $connTopK = $env:DAILY_CONNECTED_REALISM_TOPK
  if ($null -eq $connTopK -or $connTopK -notmatch '^\d+$') { $connTopK = '8' }
  $connSkipOt = $env:DAILY_CONNECTED_REALISM_SKIP_OT
  if ($null -eq $connSkipOt -or $connSkipOt -eq '') { $connSkipOt = '1' }
  $connQS = $env:DAILY_CONNECTED_REALISM_QSAMPLES
  if ($null -eq $connQS -or $connQS -notmatch '^\d+$') { $connQS = '' }
  $connCS = $env:DAILY_CONNECTED_REALISM_CSAMPLES
  if ($null -eq $connCS -or $connCS -notmatch '^\d+$') { $connCS = '' }

  # Optional: connected-sim model guardrails (OFF by default).
  # Recommended starting point (from sweep): alpha=0.10, max_scale=0.10
  $connGrAlpha = $env:DAILY_CONNECTED_REALISM_GUARDRAIL_ALPHA
  if ($null -eq $connGrAlpha -or $connGrAlpha -eq '') { $connGrAlpha = '0.0' }
  $connGrMax = $env:DAILY_CONNECTED_REALISM_GUARDRAIL_MAX_SCALE
  if ($null -eq $connGrMax -or $connGrMax -eq '') { $connGrMax = '0.10' }

  Write-Log ("Running connected realism (days={0}, topK={1}, skipOT={2})" -f $connDays, $connTopK, $connSkipOt)
  $plist = @('-m','nba_betting.cli','evaluate-connected-realism','--days', $connDays, '--top-k', $connTopK)
  if ($connSkipOt -match '^(1|true|yes)$') { $plist += '--skip-ot' }
  if ($connQS -ne '') { $plist += @('--n-quarter-samples', $connQS) }
  if ($connCS -ne '') { $plist += @('--n-connected-samples', $connCS) }

  try {
    $ga = [double]$connGrAlpha
    if ($ga -gt 0.0) {
      Write-Log ("Connected realism: guardrails enabled (alpha={0}, max_scale={1})" -f $connGrAlpha, $connGrMax)
      $plist += @('--guardrail-alpha', $connGrAlpha, '--guardrail-max-scale', $connGrMax)
    }
  } catch {
    # If parsing fails, keep guardrails off.
  }

  $connStarted = Get-Date
  $rcConn = Invoke-PyMod -plist $plist
  $elapsed = (Get-Date) - $connStarted
  Write-Log ("evaluate-connected-realism exit code: {0} (elapsed={1:n2}s)" -f $rcConn, $elapsed.TotalSeconds)
  return $rcConn
}

# Helper: invoke the future-safe daily-update subset already implemented in app.py.
# This avoids re-running historical reconciliation/actuals logic for Date+N while still
# producing the Render/UI-facing artifacts (predictions, smart_sim, recommendations, props).
function Invoke-LookAheadDailyUpdateJob {
  param([string]$TargetDate)

  try {
    if ($null -eq $TargetDate -or $TargetDate -eq '') { return $false }
    Write-Log ("Look-ahead: building future-safe artifacts for {0}" -f $TargetDate)

    $tmpPyLookAhead = Join-Path $LogPath ("lookahead_daily_update_{0}_{1}.py" -f $TargetDate, $Stamp)
    $pyLookAhead = @"
import sys
from pathlib import Path

repo_root = Path(r"{REPO_PLACEHOLDER}")
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import app

date_str = r"{DATE_PLACEHOLDER}"
app._daily_update_job(date_str=date_str, mode="lookahead", do_push=False)
ok = bool(app._job_state.get("ok"))
print("OK" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
"@
    $pyLookAhead = $pyLookAhead.Replace('{REPO_PLACEHOLDER}', $RepoRoot)
    $pyLookAhead = $pyLookAhead.Replace('{DATE_PLACEHOLDER}', $TargetDate)
    Set-Content -Path $tmpPyLookAhead -Value $pyLookAhead -Encoding UTF8

    $null = & $Python $tmpPyLookAhead 2>&1 | Tee-Object -FilePath $LogFile -Append
    $rcLookAhead = $LASTEXITCODE
    Write-Log ("Look-ahead pipeline exit code ({0}): {1}" -f $TargetDate, $rcLookAhead)
    if ($rcLookAhead -ne 0) { return $false }

    $required = @(
      (Join-Path $RepoRoot ("data/processed/predictions_{0}.csv" -f $TargetDate)),
      (Join-Path $RepoRoot ("data/processed/props_predictions_{0}.csv" -f $TargetDate)),
      (Join-Path $RepoRoot ("data/processed/recommendations_{0}.csv" -f $TargetDate)),
      (Join-Path $RepoRoot ("data/processed/props_recommendations_{0}.csv" -f $TargetDate))
    )
    $missing = @($required | Where-Object { -not (Test-Path $_) })
    $smartSimCount = @(
      Get-ChildItem (Join-Path $RepoRoot ("data/processed/smart_sim_{0}_*.json" -f $TargetDate)) -ErrorAction SilentlyContinue
    ).Count

    if ($missing.Count -gt 0) {
      Write-Log ("Look-ahead artifacts incomplete for {0}; missing: {1}" -f $TargetDate, ($missing -join ', '))
      return $false
    }
    if ($smartSimCount -le 0) {
      Write-Log ("Look-ahead artifacts incomplete for {0}; no smart_sim files found" -f $TargetDate)
      return $false
    }

    Write-Log ("Look-ahead artifacts ready for {0} (smart_sim_files={1})" -f $TargetDate, $smartSimCount)
    return $true
  } catch {
    Write-Log ("Look-ahead pipeline failed for {0}: {1}" -f $TargetDate, $_.Exception.Message)
    return $false
  }
}

# Enforce local-only pipeline; skip any server detection/calls
Write-Log 'Remote server calls disabled; running everything locally'

# Optional: run ONLY connected-realism evaluation and exit early.
# This is useful for lightweight monitoring smoke tests without triggering the full daily pipeline.
try {
  $onlyConn = $env:DAILY_ONLY_CONNECTED_REALISM
  $mode = $env:DAILY_MODE
  $only = $false
  if ($null -ne $onlyConn -and $onlyConn -match '^(1|true|yes)$') { $only = $true }
  if ($null -ne $mode -and $mode -match '^(connected[-_ ]?realism|connected[-_ ]?realism[-_ ]?only)$') { $only = $true }
  if ($only) {
    Write-Log 'Mode: connected-realism only (skipping full daily pipeline)'
    $rcOnly = Invoke-ConnectedRealismEval
    Write-Log 'Connected-realism only: done; exiting'
    exit $rcOnly
  }
} catch {
  Write-Log ("Connected-realism only mode failed: {0}" -f $_.Exception.Message)
  throw
}

# Always run local pipeline to produce site CSVs
Write-Log 'Running local pipeline to produce predictions/odds/props/edges/exports'

# Guard against preflight network hangs (rosters/logs/injuries/league status)
$PreflightTimeoutSeconds = 600
try {
  $to = $env:DAILY_PREFLIGHT_TIMEOUT_SEC
  if ($null -eq $to -or $to -eq '') { $to = '600' }
  try { $toInt = [int]$to } catch { $toInt = 600 }
  if ($toInt -lt 30) { $toInt = 30 }
  if ($toInt -gt 900) { $toInt = 900 }
  $PreflightTimeoutSeconds = $toInt
} catch { }

$LeagueStatusTimeoutSeconds = $PreflightTimeoutSeconds
try {
  $lsTo = $env:DAILY_BUILD_LEAGUE_STATUS_TIMEOUT_SEC
  if ($null -eq $lsTo -or $lsTo -eq '') { $lsTo = [string]$PreflightTimeoutSeconds }
  try { $lsToInt = [int]$lsTo } catch { $lsToInt = $PreflightTimeoutSeconds }
  if ($lsToInt -lt 30) { $lsToInt = 30 }
  if ($lsToInt -gt 900) { $lsToInt = 900 }
  $LeagueStatusTimeoutSeconds = $lsToInt
} catch { }

$seasonStr = $null
$rostersPath = $null
try {
  $seasonStr = Resolve-NbaSeasonString -DateValue $Date
  $rostersPath = Join-Path $RepoRoot ("data/processed/rosters_{0}.csv" -f $seasonStr)
} catch {
  Write-Log ("Season resolution failed for date={0}: {1}" -f $Date, $_.Exception.Message)
}

# 0) Ensure current season rosters are fetched/updated prior to projections
try {
  if ([string]::IsNullOrWhiteSpace($seasonStr)) {
    throw "season string unavailable for roster refresh"
  }
  if ([string]::IsNullOrWhiteSpace($rostersPath)) {
    $rostersPath = Join-Path $RepoRoot ("data/processed/rosters_{0}.csv" -f $seasonStr)
  }
  $isCi = $false
  try {
    $ga = $env:GITHUB_ACTIONS
    $ci = $env:CI
    if (($null -ne $ga -and $ga -match '^(1|true|yes)$') -or ($null -ne $ci -and $ci -match '^(1|true|yes)$')) { $isCi = $true }
  } catch { $isCi = $false }
  $forceRosterPreflight = $false
  try {
    $forceRosterPreflight = ($null -ne $env:DAILY_FORCE_ROSTER_PREFLIGHT -and $env:DAILY_FORCE_ROSTER_PREFLIGHT -match '^(1|true|yes)$')
  } catch { $forceRosterPreflight = $false }
  $hasRosterSeed = Test-CsvHasDataRows -Path $rostersPath
  $minRosterTeams = $env:DAILY_ROSTERS_MIN_TEAM_COUNT
  if ($null -eq $minRosterTeams -or $minRosterTeams -eq '') { $minRosterTeams = '30' }
  try { $minRosterTeamsInt = [int]$minRosterTeams } catch { $minRosterTeamsInt = 30 }
  if ($minRosterTeamsInt -lt 1) { $minRosterTeamsInt = 1 }
  $rosterTeamCount = Get-RosterTeamCount -Path $rostersPath
  $rosterLooksComplete = ($rosterTeamCount -ge $minRosterTeamsInt)
  $maxAgeH = $env:DAILY_ROSTERS_MAX_AGE_HOURS
  # NBA Stats rosters are relatively stable day-to-day, and the endpoint can be flaky.
  # Default to a wider freshness window to avoid repeated slow fetches.
  if ($null -eq $maxAgeH -or $maxAgeH -eq '') { $maxAgeH = '72' }
  try { $maxAgeMin = [int]([Math]::Max(0, ([double]$maxAgeH) * 60.0)) } catch { $maxAgeMin = 720 }
  if ((Test-FreshFile -Path $rostersPath -MaxAgeMinutes $maxAgeMin) -and $rosterLooksComplete) {
    Write-Log ("Rosters already fresh (<= {0}h); skipping fetch-rosters: {1}" -f $maxAgeH, $rostersPath)
  } elseif ((Test-FreshFile -Path $rostersPath -MaxAgeMinutes $maxAgeMin) -and -not $rosterLooksComplete) {
    Write-Log ("Rosters file looks fresh but incomplete ({0}/{1} teams); forcing fetch-rosters: {2}" -f $rosterTeamCount, $minRosterTeamsInt, $rostersPath)
  } elseif ($isCi -and -not $forceRosterPreflight -and -not $hasRosterSeed) {
    Write-Log ("CI preflight: no seeded roster artifact found at {0}; skipping fetch-rosters to avoid a full NBA Stats crawl. Set DAILY_FORCE_ROSTER_PREFLIGHT=1 to force a refresh." -f $rostersPath)
  } else {
    Write-Log ("Fetching team rosters for season {0}" -f $seasonStr)
    $rc0 = Invoke-PyModWithTimeout -plist @('-m','nba_betting.cli','fetch-rosters','--season', $seasonStr) -TimeoutSeconds $PreflightTimeoutSeconds -Label 'fetch_rosters'
    Write-Log ("fetch-rosters exit code: {0}" -f $rc0)
  }
  $rosterTeamCountAfter = Get-RosterTeamCount -Path $rostersPath
  Write-Log ("Rosters team coverage after preflight: {0}/{1}" -f $rosterTeamCountAfter, $minRosterTeamsInt)
} catch {
  Write-Log ("fetch-rosters error (non-fatal): {0}" -f $_.Exception.Message)
}
# 0.25) Refresh the season schedule artifact before predictions/league-status.
try {
  if ([string]::IsNullOrWhiteSpace($seasonStr)) {
    throw "season string unavailable for schedule refresh"
  }
  $scheduleTimeoutSeconds = [int]([Math]::Min(300, [Math]::Max(60, $PreflightTimeoutSeconds)))
  Write-Log ("Refreshing season schedule for {0}" -f $seasonStr)
  $rcSchedule = Invoke-PyModWithTimeout -plist @('-m','nba_betting.cli','fetch-schedule','--season', $seasonStr) -TimeoutSeconds $scheduleTimeoutSeconds -Label 'fetch_schedule'
  Write-Log ("fetch-schedule exit code: {0}" -f $rcSchedule)
} catch {
  Write-Log ("fetch-schedule error (non-fatal): {0}" -f $_.Exception.Message)
}
# 0.5) Fetch current-season player logs (used for roster sanity checks and calibration)
try {
  if ([string]::IsNullOrWhiteSpace($seasonStr)) {
    throw "season string unavailable for player logs"
  }
  $plCsv = Join-Path $RepoRoot 'data/processed/player_logs.csv'
  $plParquet = Join-Path $RepoRoot 'data/processed/player_logs.parquet'
  $maxAgeH = $env:DAILY_PLAYER_LOGS_MAX_AGE_HOURS
  if ($null -eq $maxAgeH -or $maxAgeH -eq '') { $maxAgeH = '12' }
  try { $maxAgeMin = [int]([Math]::Max(0, ([double]$maxAgeH) * 60.0)) } catch { $maxAgeMin = 720 }
  if (Test-FreshFile -Path $plCsv -MaxAgeMinutes $maxAgeMin) {
    Write-Log ("player_logs.csv already fresh (<= {0}h); skipping fetch-player-logs" -f $maxAgeH)
  } else {
    Write-Log ("Fetching player logs for season {0}" -f $seasonStr)
    $rcLogs = Invoke-PyModWithTimeout -plist @('-m','nba_betting.cli','fetch-player-logs','--seasons', $seasonStr) -TimeoutSeconds $PreflightTimeoutSeconds -Label 'fetch_player_logs'
    Write-Log ("fetch-player-logs exit code: {0}" -f $rcLogs)
    $playerLogsReady = $false
    if (Test-CsvHasDataRows -Path $plCsv) {
      $playerLogsReady = $true
    } elseif (Test-Path $plParquet) {
      try {
        $playerLogsReady = ((Get-Item $plParquet).Length -gt 0)
      } catch {
        $playerLogsReady = $false
      }
    }
    if (-not $playerLogsReady) {
      throw ("fetch-player-logs did not produce a usable player_logs artifact (exit={0})" -f $rcLogs)
    }
  }
} catch {
  Write-Log ("Player logs prerequisite failed: {0}" -f $_.Exception.Message)
  throw
}

# 0.55) One-time playoff transition retune after the regular season completes.
try {
  $playoffRetune = Get-PlayoffRetuneDecision -DateValue $Date -SeasonValue $seasonStr -RepoRootPath $RepoRoot
  Write-Log $playoffRetune.Reason
  if ($playoffRetune.Run) {
    $playoffRetuneTimeoutSeconds = 1800
    try {
      $prtTo = $env:DAILY_PLAYOFF_RETUNE_TIMEOUT_SEC
      if ($null -ne $prtTo -and $prtTo -ne '') {
        $playoffRetuneTimeoutSeconds = [int]$prtTo
      }
    } catch { $playoffRetuneTimeoutSeconds = 1800 }
    if ($playoffRetuneTimeoutSeconds -lt 300) { $playoffRetuneTimeoutSeconds = 300 }
    if ($playoffRetuneTimeoutSeconds -gt 7200) { $playoffRetuneTimeoutSeconds = 7200 }

    $retuneArgs = @('-m','nba_betting.cli','playoff-retune','--date', $Date)
    if (-not [string]::IsNullOrWhiteSpace($seasonStr)) {
      $retuneArgs += @('--season', $seasonStr)
    }
    $rcPlayoffRetune = Invoke-PyModWithTimeout -plist $retuneArgs -TimeoutSeconds $playoffRetuneTimeoutSeconds -Label 'playoff_retune'
    Write-Log ("playoff-retune exit code: {0}" -f $rcPlayoffRetune)
    if ($rcPlayoffRetune -ne 0) {
      if ($null -ne $env:DAILY_FORCE_PLAYOFF_RETUNE -and $env:DAILY_FORCE_PLAYOFF_RETUNE -match '^(1|true|yes)$') {
        throw ("playoff-retune failed with exit code {0}" -f $rcPlayoffRetune)
      }
      Write-Log ("WARNING: playoff-retune failed (exit={0}); continuing with existing models/artifacts" -f $rcPlayoffRetune)
    }
  }
} catch {
  if ($null -ne $env:DAILY_FORCE_PLAYOFF_RETUNE -and $env:DAILY_FORCE_PLAYOFF_RETUNE -match '^(1|true|yes)$') {
    Write-Log ("Playoff retune prerequisite failed: {0}" -f $_.Exception.Message)
    throw
  }
  Write-Log ("Playoff retune error (non-fatal): {0}" -f $_.Exception.Message)
}

# 0.6) Trade-deadline hardening: fetch injuries + build league_status + validate expected dressed players
# This must run BEFORE any predictions/sims so the player pool is up-to-date.
try {
  $injPath = Join-Path $RepoRoot 'data/raw/injuries.csv'
  $maxAgeMin = $env:DAILY_INJURIES_MAX_AGE_MINUTES
  if ($null -eq $maxAgeMin -or $maxAgeMin -eq '') { $maxAgeMin = '30' }
  try { $maxAgeMinInt = [int]$maxAgeMin } catch { $maxAgeMinInt = 30 }
  if (Test-FreshFile -Path $injPath -MaxAgeMinutes $maxAgeMinInt) {
    Write-Log ("injuries.csv already fresh (<= {0}m); skipping fetch-injuries" -f $maxAgeMinInt)
  } else {
    Write-Log "Fetching injuries from NBA official (availability gate; fallback to Rotowire/ESPN)"
    $rcInjEarly = Invoke-PyModWithTimeout -plist @('-m','nba_betting.cli','fetch-injuries','--date', $Date) -TimeoutSeconds $PreflightTimeoutSeconds -Label 'fetch_injuries'
    Write-Log ("fetch-injuries exit code: {0}" -f $rcInjEarly)
  }
} catch { Write-Log ("fetch-injuries error (non-fatal): {0}" -f $_.Exception.Message) }

try {
  $lsPath = Join-Path $RepoRoot ("data/processed/league_status_{0}.csv" -f $Date)
  $maxAgeMin = $env:DAILY_LEAGUE_STATUS_MAX_AGE_MINUTES
  if ($null -eq $maxAgeMin -or $maxAgeMin -eq '') { $maxAgeMin = '60' }
  try { $maxAgeMinInt = [int]$maxAgeMin } catch { $maxAgeMinInt = 60 }
  if (Test-FreshFile -Path $lsPath -MaxAgeMinutes $maxAgeMinInt) {
    Write-Log ("league_status already fresh (<= {0}m); skipping build-league-status" -f $maxAgeMinInt)
  } else {
    Write-Log "Building league_status for today's slate (availability gate)"
    $rcLSEarly = Invoke-PyModWithTimeout -plist @('-m','nba_betting.cli','build-league-status','--date', $Date) -TimeoutSeconds $LeagueStatusTimeoutSeconds -Label 'build_league_status'
    Write-Log ("build-league-status exit code: {0}" -f $rcLSEarly)
  }
} catch { Write-Log ("build-league-status failed (non-fatal): {0}" -f $_.Exception.Message) }

# Ensure league_status artifact exists (roster-sanity depends on it). If missing, retry once
# with a longer timeout, then fail-fast with a clear message.
try {
  $lsPath = Join-Path $RepoRoot ("data\processed\league_status_{0}.csv" -f $Date)
  if (-not (Test-Path $lsPath)) {
    Write-Log ("league_status missing after build-league-status: {0}" -f $lsPath)
    $retryTo = [int]([Math]::Min(900, [Math]::Max([Math]::Max($PreflightTimeoutSeconds * 2, $LeagueStatusTimeoutSeconds), 120)))
    Write-Log ("Retrying build-league-status with timeout {0}s" -f $retryTo)
    $rcLSRetry = Invoke-PyModWithTimeout -plist @('-m','nba_betting.cli','build-league-status','--date', $Date) -TimeoutSeconds $retryTo -Label 'build_league_status_retry'
    Write-Log ("build-league-status retry exit code: {0}" -f $rcLSRetry)
    if (-not (Test-Path $lsPath)) {
      throw ("build-league-status did not produce league_status file: {0} (exit={1})" -f $lsPath, $rcLSRetry)
    }
  }
} catch {
  Write-Log ("League status prerequisite failed: {0}" -f $_.Exception.Message)
  throw
}

# 0.65) Roster sanity check: validates slate-team roster depth + duplicates + basic team mapping.
try {
  Write-Log "Roster sanity check (fail-fast)"
  $rcRS = Invoke-PyMod -plist @('-m','nba_betting.cli','roster-sanity','--date', $Date)
  Write-Log ("roster-sanity exit code: {0}" -f $rcRS)
  if ($rcRS -ne 0) { throw "roster-sanity failed (exit=$rcRS)" }
} catch {
  Write-Log ("Roster sanity gate failed: {0}" -f $_.Exception.Message)
  throw
}

try {
  if ($NoSlateDay) {
    Write-Log ("Checking expected dressed players skipped: no slate for {0}" -f $Date)
  } else {
    Write-Log "Checking expected dressed players (fail-fast)"
    $rcDress = Invoke-PyMod -plist @('-m','nba_betting.cli','check-dressed','--date', $Date)
    Write-Log ("check-dressed exit code: {0}" -f $rcDress)
    if ($rcDress -ne 0) {
      $dressSummaryPath = Join-Path $RepoRoot ("data/processed/dressed_summary_{0}.json" -f $Date)
      $dressIssues = @()
      try {
        if (Test-Path $dressSummaryPath) {
          $dressSummary = Get-Content -Path $dressSummaryPath -Raw | ConvertFrom-Json
          if ($null -ne $dressSummary -and $null -ne $dressSummary.issues) {
            $dressIssues = @($dressSummary.issues | ForEach-Object { [string]$_ })
          }
        }
      } catch {
        $dressIssues = @()
      }

      if ($dressIssues -contains 'slate_empty') {
        $NoSlateDay = $true
        Write-Log ("check-dressed reported slate_empty for {0}; treating as a no-slate day and continuing" -f $Date)
      } else {
        throw "check-dressed failed (exit=$rcDress)"
      }
    }
  }
} catch {
  Write-Log ("Dressed-to-play gate failed: {0}" -f $_.Exception.Message)
  throw
}
# Optional: Report roster overrides applied
try {
  $ovDir = Join-Path $RepoRoot 'data/overrides'
  if (-not (Test-Path $ovDir)) { New-Item -ItemType Directory -Path $ovDir -Force | Out-Null }
  $ovPath = Join-Path $ovDir 'roster_overrides.csv'
  if (Test-Path $ovPath) {
    $ovCount = 0
    try {
      $ovCount = (Import-Csv -Path $ovPath).Count
    } catch { $ovCount = 0 }
    Write-Log ("Roster overrides present: {0} rows" -f $ovCount)
  } else {
    Write-Log "No roster overrides file found (optional)"
  }
} catch { }

# 0.66) Roster correctness audit for today (fail before predictions/exports)
try {
  $skipRosterAud = $env:DAILY_SKIP_ROSTER_AUDIT
  if ($null -eq $skipRosterAud -or $skipRosterAud -notmatch '^(1|true|yes)$') {
    Write-Log ("Auditing rosters/team assignments for {0}" -f $Date)

    $strict = $env:DAILY_STRICT_ROSTERS
    $isStrict = ($null -ne $strict -and $strict -match '^(1|true|yes)$')
    $isCi = $false
    try {
      $ga = $env:GITHUB_ACTIONS
      $ci = $env:CI
      if (($null -ne $ga -and $ga -match '^(1|true|yes)$') -or ($null -ne $ci -and $ci -match '^(1|true|yes)$')) { $isCi = $true }
    } catch { $isCi = $false }
    $warnOnly = ($isCi -and -not $isStrict)

    $lsAuditPath = Join-Path $RepoRoot ("data\processed\league_status_{0}.csv" -f $Date)
    if (-not (Test-Path $lsAuditPath)) {
      if ($warnOnly) {
        Write-Log ("WARNING: roster audit prerequisite missing; skipping roster correctness audit on CI: {0}" -f $lsAuditPath)
      } else {
        throw ("Roster correctness audit prerequisite missing: {0}" -f $lsAuditPath)
      }
    } elseif ([string]::IsNullOrWhiteSpace($rostersPath) -or -not (Test-Path $rostersPath)) {
      if ($warnOnly) {
        Write-Log ("WARNING: roster audit roster artifact missing after preflight fetch; skipping extra fetch-rosters call on CI: {0}" -f $rostersPath)
      } else {
        throw ("Roster correctness audit prerequisite missing: {0}" -f $rostersPath)
      }
    } else {
      $auditArgs = @('tools/audit_rosters_today.py','--date', $Date, '--max-mismatches', '0')
      if (-not [string]::IsNullOrWhiteSpace($seasonStr)) {
        $auditArgs += @('--season', $seasonStr)
      }
      if (-not $warnOnly) {
        $auditArgs += '--fail-if-stale'
      } else {
        Write-Log 'CI warn-only mode: running roster correctness audit without stale-roster refresh retries'
      }

      $rcRoster = Invoke-PyMod -plist $auditArgs
      Write-Log ("audit_rosters_today exit code: {0}" -f $rcRoster)

      if ($rcRoster -eq 2) {
        throw "Roster correctness audit found team mismatches (exit=$rcRoster)"
      } elseif ($rcRoster -eq 4) {
        if ($warnOnly) {
          Write-Log 'WARNING: roster correctness audit reports stale rosters; continuing on CI without re-fetching rosters (set DAILY_STRICT_ROSTERS=1 to fail)'
        } else {
          throw "Roster correctness audit reports stale rosters (exit=$rcRoster)"
        }
      } elseif ($rcRoster -ne 0) {
        if ($warnOnly) {
          Write-Log ("WARNING: roster correctness audit failed (exit={0}); continuing on CI without extra roster refreshes (set DAILY_STRICT_ROSTERS=1 to fail)" -f $rcRoster)
        } else {
          throw "Roster correctness audit failed (exit=$rcRoster)"
        }
      }
    }
  } else {
    Write-Log 'Skipping roster correctness audit (DAILY_SKIP_ROSTER_AUDIT=1)'
  }
} catch {
  Write-Log ("Roster correctness audit failed: {0}" -f $_.Exception.Message)
  throw
}

# 1) Predictions for the target date (writes data/processed/predictions_<date>.csv and may save odds)
# NOTE: --use-npu flag available but requires sklearn in NPU environment (currently blocked on ARM64 Windows)
$predictionsPath = Join-Path $RepoRoot ("data/processed/predictions_{0}.csv" -f $Date)
$gamesPredictionsNpuPath = Join-Path $RepoRoot ("data/processed/games_predictions_npu_{0}.csv" -f $Date)
$usedPredictGamesNpuFallback = $false
$predictStarted = Get-Date
$predictionsWriteTimeBefore = $null
if (Test-Path $predictionsPath) {
  try { $predictionsWriteTimeBefore = (Get-Item $predictionsPath).LastWriteTime } catch { $predictionsWriteTimeBefore = $null }
}
$rc1 = Invoke-PyMod -plist @('-m','nba_betting.cli','predict-date','--date', $Date)
Write-Log ("predict-date exit code: {0}" -f $rc1)

$predictionsRefreshed = $false
if (Test-Path $predictionsPath) {
  try {
    $predictionsWriteTimeAfter = (Get-Item $predictionsPath).LastWriteTime
    if (($null -eq $predictionsWriteTimeBefore) -or ($predictionsWriteTimeAfter -gt $predictionsWriteTimeBefore) -or ($predictionsWriteTimeAfter -ge $predictStarted.AddSeconds(-1))) {
      $predictionsRefreshed = $true
    }
  } catch {
    $predictionsRefreshed = $true
  }
}

if (-not $NoSlateDay -and -not $predictionsRefreshed) {
  Write-Log ("predict-date did not produce a fresh predictions file; attempting predict-games-npu fallback for {0}" -f $Date)
  try {
    $rcPredFallback = Invoke-PyMod -plist @('-m','nba_betting.cli','predict-games-npu','--date', $Date)
    Write-Log ("predict-games-npu fallback exit code: {0}" -f $rcPredFallback)
    if (($rcPredFallback -eq 0) -and (Test-Path $gamesPredictionsNpuPath)) {
      Copy-Item -Path $gamesPredictionsNpuPath -Destination $predictionsPath -Force
      $predictionsRefreshed = $true
      $usedPredictGamesNpuFallback = $true
      Write-Log ("predict-date fallback applied from {0} -> {1}" -f $gamesPredictionsNpuPath, $predictionsPath)
    }
  } catch {
    Write-Log ("predict-games-npu fallback failed (non-fatal): {0}" -f $_.Exception.Message)
  }
}

if ($rc1 -ne 0) {
  if ($NoSlateDay) {
    Write-Log ("WARNING: predict-date failed on a no-slate day (exit={0}); continuing without requiring refreshed predictions" -f $rc1)
  } elseif (-not $predictionsRefreshed) {
    throw ("predict-date failed and did not refresh predictions file: {0} (exit={1})" -f $predictionsPath, $rc1)
  } elseif ($usedPredictGamesNpuFallback) {
    Write-Log ("WARNING: predict-date exited nonzero; continuing with predict-games-npu fallback predictions: {0}" -f $predictionsPath)
  } else {
    Write-Log ("WARNING: predict-date exited nonzero but refreshed predictions file: {0}" -f $predictionsPath)
  }
} elseif (-not $NoSlateDay -and -not (Test-Path $predictionsPath)) {
  throw ("predict-date reported success but predictions file is missing: {0}" -f $predictionsPath)
} elseif (-not $NoSlateDay -and -not $predictionsRefreshed) {
  throw ("predict-date completed without producing a fresh predictions file: {0}" -f $predictionsPath)
} elseif (-not $predictionsRefreshed) {
  Write-Log ("WARNING: predict-date did not appear to refresh predictions file: {0}" -f $predictionsPath)
}

# Always write standardized game odds via CLI (OddsAPI consensus + Bovada fill), including prices
try {
  Write-Log "Writing game odds via CLI odds-snapshots (includes prices, prefers OddsAPI)"
  $skipOddsSnap = $env:DAILY_SKIP_ODDS_SNAPSHOTS
  if ($null -ne $skipOddsSnap -and $skipOddsSnap -match '^(1|true|yes)$') {
    Write-Log 'Skipping odds-snapshots (DAILY_SKIP_ODDS_SNAPSHOTS=1)'
  } else {
    $to = $env:DAILY_ODDS_SNAPSHOTS_TIMEOUT_SEC
    if ($null -eq $to -or $to -eq '') { $to = '180' }
    try { $toInt = [int]$to } catch { $toInt = 180 }
    if ($toInt -lt 30) { $toInt = 30 }
    if ($toInt -gt 900) { $toInt = 900 }
    $rcOdds = Invoke-PyModWithTimeout -plist @('-m','nba_betting.cli','odds-snapshots','--date', $Date) -TimeoutSeconds $toInt -Label 'odds_snapshots'
    Write-Log ("odds-snapshots exit code: {0}" -f $rcOdds)
  }
} catch { Write-Log ("odds-snapshots block failed: {0}" -f $_.Exception.Message) }

try {
  Repair-LeagueStatusFromGameOdds -TargetDate $Date -TimeoutSeconds $LeagueStatusTimeoutSeconds
} catch {
  Write-Log ("League-status post-odds repair failed: {0}" -f $_.Exception.Message)
  throw
}

# 1.5) NPU game predictions using enhanced features (CSV-based; no parquet engine required)
try {
  if ($usedPredictGamesNpuFallback -and (Test-Path $gamesPredictionsNpuPath)) {
    Write-Log ("Skipping standalone NPU game predictions for {0}; fallback artifact already exists: {1}" -f $Date, $gamesPredictionsNpuPath)
  } else {
    Write-Log ("Running NPU game predictions for {0}" -f $Date)
    $rcNpu = Invoke-PyMod -plist @('-m','nba_betting.cli','predict-games-npu','--date', $Date)
    Write-Log ("predict-games-npu exit code: {0}" -f $rcNpu)
  }
} catch {
  Write-Log ("predict-games-npu failed (non-fatal): {0}" -f $_.Exception.Message)
}

# 2) Reconcile yesterday's games (best-effort)
try {
  $yesterday = (Get-Date ([datetime]::ParseExact($Date, 'yyyy-MM-dd', $null))).AddDays(-1).ToString('yyyy-MM-dd')
} catch {
  $fallbackDate = Resolve-NbaSlateDate
  $yesterday = ([datetime]::ParseExact($fallbackDate, 'yyyy-MM-dd', $null)).AddDays(-1).ToString('yyyy-MM-dd')
}

# On no-slate days (e.g., long breaks), reconcile against the most recent slate date.
if ($NoSlateDay -and $LastSlateDate) {
  $yesterday = $LastSlateDate
}

Write-Log ("Reconcile games for {0} via local CLI" -f $yesterday)
$rc_recon = Invoke-PyMod -plist @('-m','nba_betting.cli','reconcile-date','--date', $yesterday)
Write-Log ("reconcile-date exit code: {0}" -f $rc_recon)

# 2.0) Ensure player prop actuals reconciliation exists (writes recon_props_<date>.csv)
try {
  Write-Log ("Fetching prop actuals for {0} (writes recon_props_{0}.csv when available)" -f $yesterday)
  $rcProp = Invoke-PyModWithTimeout -plist @('-m','nba_betting.cli','fetch-prop-actuals','--date', $yesterday) -TimeoutSeconds 240 -Label 'fetch_prop_actuals'
  Write-Log ("fetch-prop-actuals exit code: {0}" -f $rcProp)
  try {
    $rp = Join-Path $RepoRoot ("data/processed/recon_props_{0}.csv" -f $yesterday)
    if (Test-Path $rp) {
      Write-Log ("recon_props present: {0}" -f $rp)
    } else {
      Write-Log ("recon_props missing after fetch-prop-actuals (non-fatal): {0}" -f $rp)
    }
  } catch { }

  try {
    $skipPlayablePropAudits = $env:DAILY_SKIP_PLAYABLE_PROP_AUDITS
    if ($null -ne $skipPlayablePropAudits -and $skipPlayablePropAudits -match '^(1|true|yes)$') {
      Write-Log 'Skipping playable prop settlement/provider coverage audits (DAILY_SKIP_PLAYABLE_PROP_AUDITS=1)'
    } else {
      $auditSeason = Resolve-NbaSeasonYear -DateValue $yesterday
      $auditTimeout = $env:DAILY_PLAYABLE_PROP_AUDIT_TIMEOUT_SEC
      if ($null -eq $auditTimeout -or $auditTimeout -eq '') { $auditTimeout = '300' }
      try { $auditTimeoutInt = [int]$auditTimeout } catch { $auditTimeoutInt = 300 }
      if ($auditTimeoutInt -lt 60) { $auditTimeoutInt = 60 }
      if ($auditTimeoutInt -gt 1800) { $auditTimeoutInt = 1800 }

      Write-Log ("Auditing playable prop provider anomalies through {0} for season {1}" -f $yesterday, $auditSeason)
      $rcPlayableProviderAudit = Invoke-PyModWithTimeout -plist @(
        '-m','nba_betting.cli','audit-playable-prop-provider-anomalies',
        '--season', [string]$auditSeason,
        '--date', $yesterday
      ) -TimeoutSeconds $auditTimeoutInt -Label 'audit_playable_prop_provider_anomalies'
      Write-Log ("audit-playable-prop-provider-anomalies exit code: {0}" -f $rcPlayableProviderAudit)
      if ($rcPlayableProviderAudit -eq 0) {
        $providerAuditPath = Join-Path $RepoRoot ("data/processed/playable_prop_provider_audit_{0}.csv" -f $yesterday)
        $providerPairRollupPath = Join-Path $RepoRoot ("data/processed/playable_prop_provider_audit_pairs_{0}.csv" -f $yesterday)
        Write-PlayablePropAuditSummary -AuditKind 'provider' -RequestedDate $yesterday -AuditPath $providerAuditPath -GroupColumn 'audit_classification' -RollupPath $providerPairRollupPath
      }

      Write-Log ("Auditing playable prop coverage gaps through {0} for season {1}" -f $yesterday, $auditSeason)
      $rcPlayableCoverageAudit = Invoke-PyModWithTimeout -plist @(
        '-m','nba_betting.cli','audit-playable-prop-coverage-gaps',
        '--season', [string]$auditSeason,
        '--date', $yesterday
      ) -TimeoutSeconds $auditTimeoutInt -Label 'audit_playable_prop_coverage_gaps'
      Write-Log ("audit-playable-prop-coverage-gaps exit code: {0}" -f $rcPlayableCoverageAudit)
      if ($rcPlayableCoverageAudit -eq 0) {
        $coverageAuditPath = Join-Path $RepoRoot ("data/processed/playable_prop_coverage_audit_{0}.csv" -f $yesterday)
        $coverageTeamRollupPath = Join-Path $RepoRoot ("data/processed/playable_prop_coverage_audit_teams_{0}.csv" -f $yesterday)
        Write-PlayablePropAuditSummary -AuditKind 'coverage' -RequestedDate $yesterday -AuditPath $coverageAuditPath -GroupColumn 'coverage_subtype' -RollupPath $coverageTeamRollupPath
      }
    }
  } catch {
    Write-Log ("playable prop audits failed (non-fatal): {0}" -f $_.Exception.Message)
  }
} catch {
  Write-Log ("fetch-prop-actuals failed (non-fatal): {0}" -f $_.Exception.Message)
}

# 1.6) Calibrate games probabilities via market blend (train over last 30 days, apply to today)
try {
  $skipBlend = $env:DAILY_SKIP_GAMES_BLEND
  if ($null -eq $skipBlend -or $skipBlend -notmatch '^(1|true|yes)$') {
    Write-Log ("Calibrating games probs via market blend (30d) -> today {0}" -f $Date)
    $tmpOut = Join-Path $LogPath ("predictions_games_blend_tmp_{0}.csv" -f $Stamp)
    $blendScript = Join-Path $RepoRoot 'tools/games_blend.py'
    $blendArgs = @($blendScript, '--train-days','30','--apply-date', $Date, '--out', $tmpOut)
    $null = & $Python @blendArgs 2>&1 | Tee-Object -FilePath $LogFile -Append
    # Validate output: ensure home_win_prob_cal exists and within [0,1]
    if (Test-Path $tmpOut) {
      $ok = $false
      try {
        $pyVal = @"
import pandas as pd, sys
p = pd.read_csv(r'$tmpOut')
if 'home_win_prob_cal' in p.columns:
  s = pd.to_numeric(p['home_win_prob_cal'], errors='coerce')
  ok = s.dropna().between(0.0, 1.0).all() and s.notna().any()
  print('OK' if ok else 'NO')
else:
  print('NO')
"@
        $out = & $Python -c $pyVal
        if ($out -match '^OK') { $ok = $true }
      } catch {}
      if ($ok) {
        $predPath = Join-Path $RepoRoot ("data/processed/predictions_{0}.csv" -f $Date)
        Copy-Item -Path $tmpOut -Destination $predPath -Force
        Remove-Item $tmpOut -Force -ErrorAction SilentlyContinue
        Write-Log "Games market blend applied -> predictions updated with home_win_prob_cal"
      } else {
        Write-Log "Games blend validation failed; keeping original predictions"
        Remove-Item $tmpOut -Force -ErrorAction SilentlyContinue
      }
    } else {
      Write-Log "Games blend wrote no output; skipping apply"
    }
  } else {
    Write-Log 'Skipping games market blend (DAILY_SKIP_GAMES_BLEND=1)'
  }
} catch {
  Write-Log ("Games market blend failed (non-fatal): {0}" -f $_.Exception.Message)
}

# 1.7) Analytical simulations for ML/ATS/TOTAL using factor adjustments
try {
  Write-Log ("Running games simulations for {0}" -f $Date)
  $rcSim = Invoke-PyMod -plist @('-m','nba_betting.cli','simulate-games','--date', $Date)
  Write-Log ("simulate-games exit code: {0}" -f $rcSim)
} catch {
  Write-Log ("simulate-games failed (non-fatal): {0}" -f $_.Exception.Message)
}

# 1.6a) Optional isotonic calibration (adds home_win_prob_iso) if sufficient recent samples
try {
  $isoSkip = $env:DAILY_SKIP_ISOTONIC
  if ($null -eq $isoSkip -or $isoSkip -notmatch '^(1|true|yes)$') {
    Write-Log ("Attempt isotonic calibration for {0} (lookback=30d, min_samples=200)" -f $Date)
    $isoScript = Join-Path $RepoRoot 'tools/calibrate_isotonic.py'
    if (Test-Path $isoScript) {
      $rcIso = & $Python $isoScript --date $Date --days 30 --min-samples 200 2>&1 | Tee-Object -FilePath $LogFile -Append
      if ($rcIso -match 'train_brier_before') { Write-Log "Isotonic calibration attempted (see log for details)" } else { Write-Log "Isotonic calibration skipped or insufficient samples" }
    } else {
      Write-Log 'Isotonic script missing; skipping'
    }
  } else {
    Write-Log 'Skipping isotonic calibration (DAILY_SKIP_ISOTONIC=1)'
  }
} catch { Write-Log ("Isotonic calibration block failed (non-fatal): {0}" -f $_.Exception.Message) }

# 1.6b) Reliability curve + HTML (60d) automation
try {
  $relCsv = Join-Path $RepoRoot 'data/processed/reliability_games.csv'
  Write-Log 'Computing reliability curve (60d)' 
  $null = Invoke-PyMod -plist @('-m','nba_betting.cli','evaluate-reliability','--days','60')
  if (Test-Path $relCsv) {
    Write-Log 'Generating reliability HTML'
    $plotScript = Join-Path $RepoRoot 'tools/plot_reliability.py'
    if (Test-Path $plotScript) { & $Python $plotScript 2>&1 | Tee-Object -FilePath $LogFile -Append | Out-Null }
  } else { Write-Log 'Reliability CSV missing after compute step' }
  # 1.6b+) Calibration comparison (raw vs blend vs market)
  $calibCmp = Join-Path $RepoRoot 'tools/calibration_compare.py'
  if (Test-Path $calibCmp) {
    Write-Log 'Computing calibration comparison (60d)'
    & $Python $calibCmp --date $Date --days 60 2>&1 | Tee-Object -FilePath $LogFile -Append | Out-Null
  }
} catch { Write-Log ("Reliability automation failed (non-fatal): {0}" -f $_.Exception.Message) }

# 1.6b++) Connected sim realism (player boxscore) evaluation
# This is for accuracy/regression monitoring. It is enabled by default; set DAILY_SKIP_CONNECTED_REALISM=1 to skip.
try {
  $null = Invoke-ConnectedRealismEval
} catch { Write-Log ("Connected realism failed (non-fatal): {0}" -f $_.Exception.Message) }

# 1.6c) Drift monitoring (reference 30d vs current 7d)
try {
  $driftScript = Join-Path $RepoRoot 'tools/drift_monitor.py'
  $skipDrift = $env:DAILY_SKIP_DRIFT
  if ($null -eq $skipDrift -or $skipDrift -notmatch '^(1|true|yes)$') {
    if (Test-Path $driftScript) {
      Write-Log 'Running drift monitor (ref=30d, cur=7d)'
      & $Python $driftScript --date $Date --ref-days 30 --cur-days 7 2>&1 | Tee-Object -FilePath $LogFile -Append | Out-Null
      $driftHtmlScript = Join-Path $RepoRoot 'tools/drift_report_html.py'
      if (Test-Path $driftHtmlScript) {
        Write-Log 'Rendering drift HTML summary'
        & $Python $driftHtmlScript --date $Date 2>&1 | Tee-Object -FilePath $LogFile -Append | Out-Null
      }
      # Produce weekly drift trend rollup (last 60 days)
      $driftWeeklyScript = Join-Path $RepoRoot 'tools/drift_weekly.py'
      if (Test-Path $driftWeeklyScript) {
        Write-Log 'Rendering weekly drift trend (60d)'
        & $Python $driftWeeklyScript 2>&1 | Tee-Object -FilePath $LogFile -Append | Out-Null
      }
    } else { Write-Log 'drift_monitor.py missing; skipping drift check' }
  } else { Write-Log 'Skipping drift monitor (DAILY_SKIP_DRIFT=1)' }
} catch { Write-Log ("Drift monitoring failed (non-fatal): {0}" -f $_.Exception.Message) }

# 1.6d) Interval estimation (spread/total)
try {
  $intScript = Join-Path $RepoRoot 'tools/interval_estimation.py'
  $skipIntervals = $env:DAILY_SKIP_INTERVALS
  if ($null -eq $skipIntervals -or $skipIntervals -notmatch '^(1|true|yes)$') {
    if (Test-Path $intScript) {
      Write-Log 'Estimating predictive intervals (ref=30d, z=1.96)'
      & $Python $intScript --date $Date --ref-days 30 --z 1.96 2>&1 | Tee-Object -FilePath $LogFile -Append | Out-Null
    } else { Write-Log 'interval_estimation.py missing; skipping intervals' }
  } else { Write-Log 'Skipping interval estimation (DAILY_SKIP_INTERVALS=1)' }
} catch { Write-Log ("Interval estimation failed (non-fatal): {0}" -f $_.Exception.Message) }

$runHistoricalMaintenance = $true
$skipHistoricalMaintenance = $env:DAILY_SKIP_HISTORICAL_MAINTENANCE
if ($null -ne $skipHistoricalMaintenance -and $skipHistoricalMaintenance -match '^(1|true|yes)$') {
  $runHistoricalMaintenance = $false
}

if ($runHistoricalMaintenance) {
# 2.2) Ensure finals CSV for yesterday (best-effort; helps UI backfill and offline environments)
try {
  Write-Log ("Export finals CSV for {0}" -f $yesterday)
  $rc_fin = Invoke-PyMod -plist @('-m','nba_betting.cli','finals-export','--date', $yesterday)
  Write-Log ("finals-export exit code: {0}" -f $rc_fin)
} catch {
  Write-Log ("finals-export failed (non-fatal): {0}" -f $_.Exception.Message)
}

# 2.3) Fetch yesterday's play-by-play logs (finals-only)
try {
  Write-Log ("Fetching PBP logs for {0} (finals only)" -f $yesterday)
  $rc_pbp = Invoke-PyMod -plist @('-m','nba_betting.cli','fetch-pbp','--date', $yesterday, '--finals-only')
  Write-Log ("fetch-pbp exit code: {0}" -f $rc_pbp)
} catch {
  Write-Log ("fetch-pbp error (non-fatal): {0}" -f $_.Exception.Message)
}

# 2.4) Fetch yesterday's boxscores (finals-only)
try {
  Write-Log ("Fetching boxscores for {0} (finals only)" -f $yesterday)
  $rc_bs = Invoke-PyMod -plist @('-m','nba_betting.cli','fetch-boxscores','--date', $yesterday, '--finals-only')
  Write-Log ("fetch-boxscores exit code: {0}" -f $rc_bs)
} catch {
  Write-Log ("fetch-boxscores error (non-fatal): {0}" -f $_.Exception.Message)
}

# 2.4b) Append yesterday's boxscores into durable history (best-effort)
try {
  Write-Log ("Updating boxscores history for {0}" -f $yesterday)
  $rc_bsh = Invoke-PyMod -plist @('-m','nba_betting.cli','update-boxscores-history','--date', $yesterday, '--finals-only')
  Write-Log ("update-boxscores-history exit code: {0}" -f $rc_bsh)
} catch {
  Write-Log ("update-boxscores-history error (non-fatal): {0}" -f $_.Exception.Message)
}

# 2.4c) Append yesterday's ESPN PBP into durable history (enables rotation priors)
try {
  Write-Log ("Updating ESPN PBP history for {0}" -f $yesterday)
  $rc_pbp_espn = Invoke-PyMod -plist @('-m','nba_betting.cli','update-pbp-espn-history','--date', $yesterday, '--finals-only')
  Write-Log ("update-pbp-espn-history exit code: {0}" -f $rc_pbp_espn)
} catch {
  Write-Log ("update-pbp-espn-history error (non-fatal): {0}" -f $_.Exception.Message)
}

# 2.4d) Refresh rotation priors from substitution events (team-level)
try {
  Write-Log 'Writing rotation priors (first bench sub-in timing)'
  $rc_rot = Invoke-PyMod -plist @('-m','nba_betting.cli','write-rotation-priors','--lookback-days', '60', '--min-games', '5')
  Write-Log ("write-rotation-priors exit code: {0}" -f $rc_rot)
} catch {
  Write-Log ("write-rotation-priors error (non-fatal): {0}" -f $_.Exception.Message)
}

# 2.4e) Build ESPN rotation stints/pairs for yesterday (writes data/processed/rotations_espn/* and history files)
try {
  $skipRot = $env:DAILY_SKIP_ROTATIONS_ESPN
  if ($null -eq $skipRot -or $skipRot -notmatch '^(1|true|yes)$') {
    Write-Log ("Updating ESPN rotations history for {0} (stints/pairs/play_context)" -f $yesterday)
    $rc_rot_hist = Invoke-PyMod -plist @('-m','nba_betting.cli','update-rotations-espn-history','--date', $yesterday, '--rate-delay', '0.25')
    Write-Log ("update-rotations-espn-history exit code: {0}" -f $rc_rot_hist)

    # Gap scan: if any stints are missing for yesterday (ESPN flakiness), retry once or twice.
    try {
      $rotRetryMax = $env:DAILY_ROTATIONS_ESPN_RETRY_MAX
      if ($null -eq $rotRetryMax -or $rotRetryMax -eq '') { $rotRetryMax = '1' }
      try { $rotRetryMax = [int]$rotRetryMax } catch { $rotRetryMax = 1 }
      if ($rotRetryMax -lt 0) { $rotRetryMax = 0 }

      $tmpPyRotScan = Join-Path $LogPath ("rotations_gap_scan_{0}.py" -f $Stamp)
      $pyRotScan = @'
import os
from pathlib import Path

repo_root = Path(os.environ.get('REPO_ROOT', '.')).resolve()
date_str = os.environ.get('DATE_STR')

rot_dir = repo_root / 'data' / 'processed' / 'rotations_espn'

def _exists_nonempty(p: Path) -> bool:
    try:
        return p.exists() and p.stat().st_size > 0
    except Exception:
        return bool(p.exists())

missing = []
try:
    from nba_betting.boxscores import _nba_gid_to_tricodes
    gid_map = _nba_gid_to_tricodes(str(date_str)) or {}
    for gid in gid_map.keys():
        gid = str(gid).strip()
        if not gid:
            continue
        hp = rot_dir / f'stints_home_{gid}.csv'
        ap = rot_dir / f'stints_away_{gid}.csv'
        if not (_exists_nonempty(hp) and _exists_nonempty(ap)):
            missing.append(gid)
except Exception:
    missing = []

print('MISSING_COUNT:' + str(len(missing)))
print('MISSING_GIDS:' + ','.join(missing))
'@

      Set-Content -Path $tmpPyRotScan -Value $pyRotScan -Encoding UTF8
      $env:REPO_ROOT = $RepoRoot
      $env:DATE_STR = $yesterday

      for ($attempt = 0; $attempt -le $rotRetryMax; $attempt++) {
        $scanOut = & $Python $tmpPyRotScan 2>&1 | Tee-Object -FilePath $LogFile -Append
        $missingCount = 0
        $missingGids = ''
        if ($scanOut -match 'MISSING_COUNT:(\d+)') {
          $missingCount = [int]([regex]::Match($scanOut, 'MISSING_COUNT:(\d+)').Groups[1].Value)
        }
        if ($scanOut -match 'MISSING_GIDS:([^\r\n]*)') {
          $missingGids = [regex]::Match($scanOut, 'MISSING_GIDS:([^\r\n]*)').Groups[1].Value
        }

        if ($missingCount -le 0) {
          if ($attempt -gt 0) { Write-Log 'Rotations gap scan: OK after retry' }
          break
        }
        if ($attempt -ge $rotRetryMax) {
          Write-Log ("Rotations gap scan: still missing after retries ({0}): {1}" -f $missingCount, $missingGids)
          break
        }

        Write-Log ("Rotations gap scan: missing {0} games; retrying update-rotations-espn-history (attempt {1}/{2}) gids={3}" -f $missingCount, ($attempt + 1), $rotRetryMax, $missingGids)
        $rc_rot_retry = Invoke-PyMod -plist @('-m','nba_betting.cli','update-rotations-espn-history','--date', $yesterday, '--rate-delay', '0.5')
        Write-Log ("update-rotations-espn-history retry exit code: {0}" -f $rc_rot_retry)
      }
    } catch {
      Write-Log ("Rotations gap scan failed (non-fatal): {0}" -f $_.Exception.Message)
    }
  } else {
    Write-Log 'Skipping update-rotations-espn-history (DAILY_SKIP_ROTATIONS_ESPN=1)'
  }
} catch {
  Write-Log ("update-rotations-espn-history error (non-fatal): {0}" -f $_.Exception.Message)
}

# 2.5) Backfill missing recon files for the season to date (idempotent)
try {
  $seasonStart = '2025-10-21'
  Write-Log ("Recon backfill scan from {0} to {1}" -f $seasonStart, $yesterday)
  $d0 = [datetime]::ParseExact($seasonStart, 'yyyy-MM-dd', $null)
  $d1 = [datetime]::ParseExact($yesterday, 'yyyy-MM-dd', $null)
  $cur = $d0
  $toBuild = @()
  while ($cur -le $d1) {
    $ds = $cur.ToString('yyyy-MM-dd')
    $p = Join-Path $RepoRoot ("data/processed/recon_games_{0}.csv" -f $ds)
    if (-not (Test-Path $p)) { $toBuild += $ds }
    $cur = $cur.AddDays(1)
  }
  if ($toBuild.Count -gt 0) {
    Write-Log ("Recon backfill missing dates: {0}" -f ($toBuild -join ', '))
    Remove-StaleGitLock
    $built = @()
    foreach ($ds in $toBuild) {
      Write-Log ("Build recon for {0}" -f $ds)
      $rc_bf = Invoke-PyMod -plist @('-m','nba_betting.cli','reconcile-date','--date', $ds)
      Write-Log ("reconcile-date ({0}) exit code: {1}" -f $ds, $rc_bf)
      $pp = Join-Path $RepoRoot ("data/processed/recon_games_{0}.csv" -f $ds)
      if ($rc_bf -eq 0 -and (Test-Path $pp)) { $built += $pp }
    }
    if ($built.Count -gt 0) {
      try {
        foreach ($bf in $built) { & git add -- $bf 2>&1 | Tee-Object -FilePath $LogFile -Append | Out-Null }
        $changedBf = & git diff --cached --name-only -- data/processed/recon_games_*.csv
        if ($changedBf) {
          $msgBf = "data(processed): backfill recon games ($seasonStart..$yesterday)"
          Remove-StaleGitLock
          & git commit -m $msgBf 2>&1 | Tee-Object -FilePath $LogFile -Append | Out-Null
          $rcReconPull = Invoke-LoggedNativeCommand -FilePath 'git' -ArgumentList @('pull', '--rebase')
          if ($rcReconPull -ne 0) {
            throw "git pull --rebase failed (exit=$rcReconPull)"
          }
          $rcReconPush = Invoke-LoggedNativeCommand -FilePath 'git' -ArgumentList @('push')
          if ($rcReconPush -ne 0) {
            throw "git push failed (exit=$rcReconPush)"
          }
          Write-Log 'Git: pushed recon backfill commit'
        } else {
          Write-Log 'Git: no recon backfill changes to push'
        }
      } catch { Write-Log ("Git push (recon backfill) failed: {0}" -f $_.Exception.Message) }
    }
  } else {
    Write-Log 'Recon backfill: no missing dates'
  }
} catch { Write-Log ("Recon backfill block failed: {0}" -f $_.Exception.Message) }

# 2.4a) Reconcile PBP-derived markets for yesterday (tip, first-basket, early-threes)
try {
  Write-Log ("Reconciling PBP markets for {0}" -f $yesterday)
  $rc_pbp_recon = Invoke-PyMod -plist @('-m','nba_betting.cli','reconcile-pbp-markets','--date', $yesterday)
  Write-Log ("reconcile-pbp-markets exit code: {0}" -f $rc_pbp_recon)
} catch {
  Write-Log ("reconcile-pbp-markets failed (non-fatal): {0}" -f $_.Exception.Message)
}

# 2.4b) Update calibration for PBP markets using rolling window
try {
  Write-Log ("Calibrating PBP markets from reconciliation (window=7) anchored at {0}" -f $yesterday)
  $rc_pbp_cal = Invoke-PyMod -plist @('-m','nba_betting.cli','calibrate-pbp-markets','--anchor', $yesterday, '--window', '7')
  Write-Log ("calibrate-pbp-markets exit code: {0}" -f $rc_pbp_cal)
} catch {
  Write-Log ("calibrate-pbp-markets failed (non-fatal): {0}" -f $_.Exception.Message)
}

# 2.4d) Build per-player reconciliation + live player lens tuning dataset for yesterday
try {
  Write-Log ("Building recon_players + live_player_lens_tuning for {0}" -f $yesterday)

  # recon_players_<date>.csv (SmartSim player means vs actual)
  try {
    $null = & $Python (Join-Path $RepoRoot 'tools/build_recon_players.py') --date $yesterday 2>&1 | Tee-Object -FilePath $LogFile -Append
    Write-Log "build_recon_players completed (non-fatal)"
  } catch {
    Write-Log ("build_recon_players failed (non-fatal): {0}" -f $_.Exception.Message)
  }

  # live_player_lens_tuning_<date>.csv (props: sim vs line vs actual)
  try {
    $null = & $Python (Join-Path $RepoRoot 'tools/build_live_player_lens_tuning.py') --date $yesterday 2>&1 | Tee-Object -FilePath $LogFile -Append
    Write-Log "build_live_player_lens_tuning completed (non-fatal)"
  } catch {
    Write-Log ("build_live_player_lens_tuning failed (non-fatal): {0}" -f $_.Exception.Message)
  }
} catch {
  Write-Log ("Recon players / tuning block failed (non-fatal): {0}" -f $_.Exception.Message)
}

# 2.4c) Log daily PBP market health metrics for yesterday (CSV-first)
try {
  Write-Log ("Logging daily PBP market metrics for {0}" -f $yesterday)
  $pyMetrics = @'
import os, pandas as pd, numpy as np

root = os.environ.get("ROOT")
date = os.environ.get("DATE")
out = os.environ.get("OUT")
if not (root and date and out):
    print("NO:missing_env"); raise SystemExit(0)
rec = os.path.join(root, "data", "processed", f"pbp_reconcile_{date}.csv")
if not os.path.exists(rec):
    print("NO:missing_reconcile"); raise SystemExit(0)
df = pd.read_csv(rec)

def _mean(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.mean()) if len(s) > 0 else float("nan")

tip_brier = _mean(df.get("tip_brier"))
tip_logloss = _mean(df.get("tip_logloss"))
tip_prob = pd.to_numeric(df.get("tip_prob_home"), errors="coerce") if "tip_prob_home" in df.columns else pd.Series(dtype=float)
tip_out = pd.to_numeric(df.get("tip_outcome_home"), errors="coerce") if "tip_outcome_home" in df.columns else pd.Series(dtype=float)
tip_acc = float(np.mean([(ph >= 0.5) == (oh == 1.0) for ph, oh in zip(tip_prob.dropna(), tip_out.dropna())])) if (len(tip_prob.dropna()) > 0 and len(tip_out.dropna()) > 0) else float("nan")
tip_n = int(pd.notna(df.get("tip_brier")).sum()) if "tip_brier" in df.columns else 0

fb_hit1 = pd.to_numeric(df.get("first_basket_hit_top1"), errors="coerce").dropna() if "first_basket_hit_top1" in df.columns else pd.Series(dtype=float)
fb_hit5 = pd.to_numeric(df.get("first_basket_hit_top5"), errors="coerce").dropna() if "first_basket_hit_top5" in df.columns else pd.Series(dtype=float)
fb_prob_act = pd.to_numeric(df.get("first_basket_prob_actual"), errors="coerce").dropna() if "first_basket_prob_actual" in df.columns else pd.Series(dtype=float)
fb_top1 = float(fb_hit1.mean()) if len(fb_hit1) > 0 else float("nan")
fb_top5 = float(fb_hit5.mean()) if len(fb_hit5) > 0 else float("nan")
fb_mean_prob_actual = float(fb_prob_act.mean()) if len(fb_prob_act) > 0 else float("nan")
fb_n = int(max(len(fb_hit1), len(fb_hit5))) if (len(fb_hit1) > 0 or len(fb_hit5) > 0) else 0

thr_err = pd.to_numeric(df.get("early_threes_error"), errors="coerce").dropna() if "early_threes_error" in df.columns else pd.Series(dtype=float)
thr_mae = float(thr_err.abs().mean()) if len(thr_err) > 0 else float("nan")
thr_rmse = float(np.sqrt((thr_err ** 2).mean())) if len(thr_err) > 0 else float("nan")
thr_brier = _mean(df.get("early_threes_brier_ge1"))
thr_n = int(len(thr_err))

row = {
    "date": date,
    "tip_n": tip_n, "tip_brier": tip_brier, "tip_logloss": tip_logloss, "tip_acc": tip_acc,
    "fb_n": fb_n, "fb_top1": fb_top1, "fb_top5": fb_top5, "fb_mean_prob_actual": fb_mean_prob_actual,
    "thr_n": thr_n, "thr_mae": thr_mae, "thr_rmse": thr_rmse, "thr_brier_ge1": thr_brier,
}

cols = list(row.keys())
if os.path.exists(out):
    try:
        ex = pd.read_csv(out)
        if not ex.empty and "date" in ex.columns:
            ex = ex[ex["date"].astype(str) != str(date)]
        ex = pd.concat([ex, pd.DataFrame([row])], ignore_index=True)
        ex.to_csv(out, index=False)
    except Exception:
        pd.DataFrame([row])[cols].to_csv(out, index=False)
else:
    pd.DataFrame([row])[cols].to_csv(out, index=False)
print("OK")
'@
  $metricsYear = ($yesterday.Substring(0,4))
  $env:ROOT = $RepoRoot
  $env:DATE = $yesterday
  $env:OUT = (Join-Path $RepoRoot ("data/processed/pbp_metrics_daily_{0}.csv" -f $metricsYear))
  $tmpPyM = Join-Path $LogPath ("pbp_metrics_daily_{0}.py" -f $Stamp)
  Set-Content -Path $tmpPyM -Value $pyMetrics -Encoding UTF8
  $outM = & $Python $tmpPyM 2>&1 | Tee-Object -FilePath $LogFile -Append
  if ($outM -match 'OK') {
    Write-Log ("Logged PBP metrics -> {0}" -f $env:OUT)
  } else {
    Write-Log ("PBP metrics logging returned: {0}" -f $outM)
  }
} catch {
  Write-Log ("PBP metrics logging failed (non-fatal): {0}" -f $_.Exception.Message)
}

# 2.4c.pre) Ensure PBP inputs exist for recent days to support recon_quarters backfill (7 days)
try {
  $start = (Get-Date ([datetime]::ParseExact($yesterday, 'yyyy-MM-dd', $null))).AddDays(-7).ToString('yyyy-MM-dd')
  $end = $yesterday
  Write-Log ("Backfilling PBP for {0}..{1} (finals-only, last 7d) to enable recon_quarters" -f $start, $end)
  $rc_bfpbp = Invoke-PyMod -plist @('-m','nba_betting.cli','backfill-pbp','--start', $start, '--end', $end, '--finals-only')
  Write-Log ("backfill-pbp exit code: {0}" -f $rc_bfpbp)
} catch {
  Write-Log ("backfill-pbp failed (non-fatal): {0}" -f $_.Exception.Message)
}

# 2.4d) Reconcile quarters/halves vs predictions for yesterday
try {
  Write-Log ("Reconciling quarters for {0}" -f $yesterday)
  $rc_qrecon = Invoke-PyMod -plist @('-m','nba_betting.cli','reconcile-quarters','--date', $yesterday)
  Write-Log ("reconcile-quarters exit code: {0}" -f $rc_qrecon)
  $reconQuartersYesterdayPath = Join-Path $RepoRoot ("data/processed/recon_quarters_{0}.csv" -f $yesterday)
  if (($rc_qrecon -eq 0) -and (-not (Test-Path $reconQuartersYesterdayPath))) {
    Write-Log ("WARNING: reconcile-quarters reported success but wrote no file: {0}" -f $reconQuartersYesterdayPath)
  }
} catch {
  Write-Log ("reconcile-quarters failed (non-fatal): {0}" -f $_.Exception.Message)
}

# 2.4d.a) Backfill recent recon_quarters before yesterday to seed calibration
try {
  $start = (Get-Date ([datetime]::ParseExact($yesterday, 'yyyy-MM-dd', $null))).AddDays(-6)
  $end = ([datetime]::ParseExact($yesterday, 'yyyy-MM-dd', $null)).AddDays(-1)
  $cur = $start
  $missing = @()
  while ($cur -le $end) {
    $ds = $cur.ToString('yyyy-MM-dd')
    $p = Join-Path $RepoRoot ("data/processed/recon_quarters_{0}.csv" -f $ds)
    if (-not (Test-Path $p)) { $missing += $ds }
    $cur = $cur.AddDays(1)
  }
  if ($missing.Count -gt 0) {
    Write-Log ("Recon quarters backfill (previous 6d) missing: {0}" -f ($missing -join ', '))
    foreach ($ds in $missing) {
      try {
        Write-Log ("Build recon-quarters for {0}" -f $ds)
        $rc_rq = Invoke-PyMod -plist @('-m','nba_betting.cli','reconcile-quarters','--date', $ds)
        Write-Log ("reconcile-quarters ({0}) exit code: {1}" -f $ds, $rc_rq)
      } catch { Write-Log ("reconcile-quarters ({0}) failed: {1}" -f $ds, $_.Exception.Message) }
    }
  } else {
    Write-Log 'Recon quarters backfill: none missing in previous 6 days'
  }
} catch { Write-Log ("Recon quarters backfill block failed (non-fatal): {0}" -f $_.Exception.Message) }

# 2.4d.b) Build quarter calibration artifact (used by SmartSim per-quarter realism)
# Controlled by env DAILY_SKIP_QUARTERS_CALIB=1; otherwise rebuild if missing, older than 7 days,
# or recon_quarters_* has been updated more recently than the calibration artifact.
try {
  $skipQCal = $env:DAILY_SKIP_QUARTERS_CALIB
  if ($null -ne $skipQCal -and $skipQCal -match '^(1|true|yes)$') {
    Write-Log 'Skipping quarters calibration build (DAILY_SKIP_QUARTERS_CALIB=1)'
  } else {
    $qcalPath = Join-Path $RepoRoot 'data/processed/quarters_calibration.json'
    $needBuild = $true
    if (Test-Path $qcalPath) {
      $needBuild = $false
      try {
        $qcalItem = Get-Item $qcalPath
        $ageDays = ((Get-Date) - $qcalItem.LastWriteTime).TotalDays
        if ($ageDays -ge 7) { $needBuild = $true }

        $reconGlob = Join-Path $RepoRoot 'data/processed/recon_quarters_*.csv'
        $latestRecon = Get-ChildItem -Path $reconGlob -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($null -ne $latestRecon -and $latestRecon.LastWriteTime -gt $qcalItem.LastWriteTime) {
          $needBuild = $true
        }
      } catch { }
    }
    if ($needBuild) {
      Write-Log 'Building quarters calibration -> data/processed/quarters_calibration.json'
      $rc_qcal = Invoke-PyMod -plist @('tools/build_quarters_calibration.py','--workspace', $RepoRoot, '--out', 'data/processed/quarters_calibration.json')
      Write-Log ("build_quarters_calibration exit code: {0}" -f $rc_qcal)
    } else {
      Write-Log 'Quarters calibration: up-to-date (skipping rebuild)'
    }
  }
} catch {
  Write-Log ("Quarters calibration build failed (non-fatal): {0}" -f $_.Exception.Message)
}

# 2.4e) Calibrate game totals (global + team) using rolling window anchored at yesterday
try {
  Write-Log ("Calibrating game totals (window=14) anchored at {0}" -f $yesterday)
  $rc_cal_tot = Invoke-PyMod -plist @('-m','nba_betting.cli','calibrate-totals','--anchor', $yesterday, '--window', '14')
  Write-Log ("calibrate-totals exit code: {0}" -f $rc_cal_tot)
} catch {
  Write-Log ("calibrate-totals failed (non-fatal): {0}" -f $_.Exception.Message)
}

# 2.4e1) Build/refresh smart_sim_quarter_eval for recent window (non-fatal)
# This is used to calibrate period probabilities (over + cover) and to feed evaluation.
# Controlled by env DAILY_SKIP_SMART_SIM_EVAL_BUILD=1
try {
  $skipSSEval = $env:DAILY_SKIP_SMART_SIM_EVAL_BUILD
  if ($null -ne $skipSSEval -and $skipSSEval -match '^(1|true|yes)$') {
    Write-Log 'Skipping smart_sim_quarter_eval build (DAILY_SKIP_SMART_SIM_EVAL_BUILD=1)'
  } else {
    # Build/refresh PBP-derived period actuals so smart_sim_quarter_eval can join actual quarter/half scores
    # even when games_nba_api.csv is missing recent seasons.
    # Controlled by env DAILY_SKIP_PBP_PERIOD_ACTUALS=1
    try {
      $skipPbpAct = $env:DAILY_SKIP_PBP_PERIOD_ACTUALS
      if ($null -ne $skipPbpAct -and $skipPbpAct -match '^(1|true|yes)$') {
        Write-Log 'Skipping PBP period actuals build (DAILY_SKIP_PBP_PERIOD_ACTUALS=1)'
      } else {
        $pbpDays = $env:DAILY_PBP_PERIOD_ACTUALS_DAYS
        if ($null -eq $pbpDays -or $pbpDays -eq '') { $pbpDays = '7' }
        try { $pbpDaysInt = [int]$pbpDays } catch { $pbpDaysInt = 7 }
        if ($pbpDaysInt -lt 1) { $pbpDaysInt = 1 }
        if ($pbpDaysInt -gt 21) { $pbpDaysInt = 21 }
        $pbpStart = ([datetime]::ParseExact($yesterday, 'yyyy-MM-dd', $null)).AddDays(-($pbpDaysInt - 1)).ToString('yyyy-MM-dd')
        Write-Log ("Building PBP period actuals (start={0}, end={1})" -f $pbpStart, $yesterday)
        $rc_pbp_act = Invoke-PyMod -plist @('tools/build_period_actuals_from_pbp_espn.py','--start', $pbpStart, '--end', $yesterday)
        Write-Log ("build_period_actuals_from_pbp_espn exit code: {0}" -f $rc_pbp_act)
      }
    } catch {
      Write-Log ("PBP period actuals build failed (non-fatal): {0}" -f $_.Exception.Message)
    }

    Write-Log ("Building smart_sim_quarter_eval (end={0}, days=60)" -f $yesterday)
    $rc_ss_eval = Invoke-PyMod -plist @('tools/build_smart_sim_quarter_eval.py','--end', $yesterday, '--days', '60')
    Write-Log ("build_smart_sim_quarter_eval exit code: {0}" -f $rc_ss_eval)
  }
} catch {
  Write-Log ("smart_sim_quarter_eval build failed (non-fatal): {0}" -f $_.Exception.Message)
}

# 2.4e2) Calibrate period (quarters/halves) over probabilities using smart_sim_quarter_eval (non-fatal)
# Controlled by env DAILY_SKIP_PERIOD_PROBS_CALIB=1
try {
  $skipPProb = $env:DAILY_SKIP_PERIOD_PROBS_CALIB
  if ($null -ne $skipPProb -and $skipPProb -match '^(1|true|yes)$') {
    Write-Log 'Skipping period probability calibration build (DAILY_SKIP_PERIOD_PROBS_CALIB=1)'
  } else {
    Write-Log ("Calibrating period over probabilities (window=30) anchored at {0}" -f $yesterday)
    $rc_cal_pprob = Invoke-PyMod -plist @('-m','nba_betting.cli','calibrate-period-probs','--anchor', $yesterday, '--window', '30', '--bins', '12', '--alpha', '1.0')
    Write-Log ("calibrate-period-probs exit code: {0}" -f $rc_cal_pprob)
  }
} catch {
  Write-Log ("calibrate-period-probs failed (non-fatal): {0}" -f $_.Exception.Message)
}

# 2.4f) Apply totals calibration to today's predictions (adjusts totals and period totals if present)
if (-not $SkipTotalsCalib) {
  try {
    Write-Log ("Applying totals calibration from {0} to predictions for {1}" -f $yesterday, $Date)

    # Back up the predictions file before applying calibration (safety valve)
    $predPath = Join-Path $RepoRoot ("data/processed/predictions_{0}.csv" -f $Date)
    $backupPath = Join-Path $RepoRoot ("data/processed/_predictions_backup_{0}.csv" -f $Date)
    if (Test-Path $predPath) {
      try { Copy-Item -Path $predPath -Destination $backupPath -Force } catch { Write-Log ("Backup create failed (non-fatal): {0}" -f $_.Exception.Message) }
    } else {
      Write-Log "No predictions CSV found prior to calibration; skipping backup"
    }

    # Apply into a temporary file first; only replace original if validation passes
    $tmpOut = Join-Path $LogPath ("predictions_calib_tmp_{0}.csv" -f $Stamp)
    $rc_apply_tot = Invoke-PyMod -plist @('-m','nba_betting.cli','apply-totals-calibration','--date', $Date, '--calib-date', $yesterday, '--in', $predPath, '--out', $tmpOut)
    Write-Log ("apply-totals-calibration exit code: {0}" -f $rc_apply_tot)

    # Validate predictions after apply; if invalid, restore backup
    try {
      $tmpPy = Join-Path $LogPath ("validate_predictions_{0}.py" -f $Stamp)
      $pyCode = @"
import sys, os, pandas as pd, numpy as np
from pathlib import Path

pred_path = Path(r"$tmpOut")
ok=True; why=[]; stats={}
if not pred_path.exists():
  print("NO:file_missing"); sys.exit(0)
df = pd.read_csv(pred_path)

# Thresholds from environment with sensible defaults
def _getf(name, default):
  try:
    v = os.environ.get(name)
    return float(v) if v is not None and str(v).strip() != '' else float(default)
  except Exception:
    return float(default)
TOT_MIN = _getf('TOTALS_MIN', 120)
TOT_MAX = _getf('TOTALS_MAX', 300)
QTR_MIN = _getf('QTR_MIN', 15)
QTR_MAX = _getf('QTR_MAX', 80)
HALF_MIN = _getf('HALF_MIN', 30)
HALF_MAX = _getf('HALF_MAX', 160)
QSUM_TOL = _getf('QSUM_TOL', 25)

def in_range(s, lo, hi):
  import pandas as _pd
  s = _pd.to_numeric(s, errors="coerce")
  if s.isna().all():
    return False
  v = s.dropna().astype(float)
  try:
    key = getattr(s, 'name', None) or 'col'
    if len(v) > 0:
      stats[key] = {'min': float(np.nanmin(v)), 'max': float(np.nanmax(v))}
  except Exception:
    pass
  return bool(((s >= lo) & (s <= hi)).fillna(True).all())

cols = set(df.columns)
if ("totals" not in cols) or (not in_range(df["totals"], TOT_MIN, TOT_MAX)):
  ok=False; why.append("totals_out_of_range")
for c, lo, hi in [
  ("quarters_q1_total", QTR_MIN, QTR_MAX),
  ("quarters_q2_total", QTR_MIN, QTR_MAX),
  ("quarters_q3_total", QTR_MIN, QTR_MAX),
  ("quarters_q4_total", QTR_MIN, QTR_MAX),
  ("halves_h1_total", HALF_MIN, HALF_MAX),
  ("halves_h2_total", HALF_MIN, HALF_MAX),
]:
  if c in cols and not in_range(df[c], lo, hi):
    ok=False; why.append(f"{c}_out_of_range")

# Optional: quarter sum approx equals game totals within tolerance where all quarters present
try:
  need = ["quarters_q1_total","quarters_q2_total","quarters_q3_total","quarters_q4_total","totals"]
  if all(n in cols for n in need):
    import pandas as _pd
    qsum = sum(_pd.to_numeric(df[n], errors="coerce") for n in need[:-1])
    tot = _pd.to_numeric(df["totals"], errors="coerce")
    diff = (qsum - tot).abs()
    if not (diff <= QSUM_TOL).fillna(True).all():
      ok=False; why.append("quarter_sum_mismatch")
except Exception:
  pass

if ok:
  print("OK:"+str({k:{'min':v['min'],'max':v['max']} for k,v in stats.items()}))
else:
  print("NO:"+",".join(why)+";stats:"+str({k:{'min':v['min'],'max':v['max']} for k,v in stats.items()}))
"@
      Set-Content -Path $tmpPy -Value $pyCode -Encoding UTF8
      $valOut = & $Python $tmpPy 2>&1 | Tee-Object -FilePath $LogFile -Append
      if ($valOut -notmatch '^OK') {
        Write-Log ("Calibration validation failed: {0}" -f $valOut)
        try { if (Test-Path $tmpOut) { Remove-Item $tmpOut -Force } } catch { }
        if ((Test-Path $backupPath) -and (Test-Path $predPath)) {
          Write-Log "Kept original predictions (pre-calibration) due to failed validation"
        }
      } else {
        Write-Log "Calibration validation: OK"
        try {
          if (Test-Path $tmpOut) {
            Copy-Item -Path $tmpOut -Destination $predPath -Force
            Remove-Item $tmpOut -Force
          }
        } catch { Write-Log ("Finalizing calibrated predictions failed: {0}" -f $_.Exception.Message) }
      }
    } catch {
      Write-Log ("Validation block failed (non-fatal): {0}" -f $_.Exception.Message)
    }
  } catch {
    Write-Log ("apply-totals-calibration failed (non-fatal): {0}" -f $_.Exception.Message)
  }
} else {
  Write-Log 'Skipping totals calibration (SkipTotalsCalib=true)'
}

} else {
  Write-Log 'Skipping historical maintenance block on this run (DAILY_SKIP_HISTORICAL_MAINTENANCE=1); prioritizing same-day artifacts.'
}

# 2.5) Roster audit for yesterday (requires boxscores); writes roster_audit_<yesterday>.csv
try {
  $skipYesterdayRosterAudit = $env:DAILY_SKIP_YESTERDAY_ROSTER_AUDIT
  $forceYesterdayRosterAudit = $env:DAILY_FORCE_YESTERDAY_ROSTER_AUDIT
  $strictYesterdayRosterAudit = $env:DAILY_STRICT_ROSTERS
  $isStrictYesterdayRosterAudit = ($null -ne $strictYesterdayRosterAudit -and $strictYesterdayRosterAudit -match '^(1|true|yes)$')
  $isCiYesterdayRosterAudit = $false
  try {
    $ga = $env:GITHUB_ACTIONS
    $ci = $env:CI
    if (($null -ne $ga -and $ga -match '^(1|true|yes)$') -or ($null -ne $ci -and $ci -match '^(1|true|yes)$')) { $isCiYesterdayRosterAudit = $true }
  } catch { $isCiYesterdayRosterAudit = $false }

  if ($null -ne $skipYesterdayRosterAudit -and $skipYesterdayRosterAudit -match '^(1|true|yes)$') {
    Write-Log 'Skipping yesterday roster audit (DAILY_SKIP_YESTERDAY_ROSTER_AUDIT=1)'
  } elseif ($isCiYesterdayRosterAudit -and -not $isStrictYesterdayRosterAudit -and -not ($null -ne $forceYesterdayRosterAudit -and $forceYesterdayRosterAudit -match '^(1|true|yes)$')) {
    Write-Log 'Skipping yesterday roster audit on CI warn-only mode (set DAILY_FORCE_YESTERDAY_ROSTER_AUDIT=1 or DAILY_STRICT_ROSTERS=1 to enable)'
  } else {
    Write-Log ("Running roster audit for {0}" -f $yesterday)
    $to = $env:DAILY_ROSTER_AUDIT_TIMEOUT_SEC
    # Default higher: nba_api boxscore pulls can be slow.
    if ($null -eq $to -or $to -eq '') { $to = '600' }
    try { $toInt = [int]$to } catch { $toInt = 600 }
    if ($toInt -lt 30) { $toInt = 30 }
    if ($toInt -gt 1800) { $toInt = 1800 }
    $rc_audit = Invoke-PyModWithTimeout -plist @('-m','nba_betting.cli','audit-rosters','--date', $yesterday) -TimeoutSeconds $toInt -Label 'audit_rosters'
    Write-Log ("audit-rosters exit code: {0}" -f $rc_audit)
  }
} catch {
  Write-Log ("audit-rosters error (non-fatal): {0}" -f $_.Exception.Message)
}

# 2.1) Finals export dedupe: only run custom export if CSV missing and CLI path didn't produce it
try {
  $finalsCsv = Join-Path $RepoRoot ("data/processed/finals_{0}.csv" -f $yesterday)
  if (-not (Test-Path $finalsCsv)) {
    Write-Log ("Exporting finals CSV for {0} (custom path)" -f $yesterday)
    $pyFinals = @'
import os
from app import _write_finals_csv_for_date
d = os.environ.get("YDAY")
if d:
    p, n = _write_finals_csv_for_date(d)
    print(f"WROTE:{p}:{n}")
else:
    print("NO_DATE")
'@
    $env:YDAY = $yesterday
    $tmpPyF = Join-Path $LogPath ("finals_export_{0}.py" -f $Stamp)
    Set-Content -Path $tmpPyF -Value $pyFinals -Encoding UTF8
    $outF = & $Python $tmpPyF 2>&1 | Tee-Object -FilePath $LogFile -Append
    if ($outF -match 'WROTE:') { Write-Log ("Finals export result: {0}" -f $outF) } else { Write-Log ("Finals export returned: {0}" -f $outF) }
  } else {
    Write-Log ("Finals CSV already present for {0}; skipping custom export" -f $yesterday)
  }
} catch { Write-Log ("Finals export block failed (non-fatal): {0}" -f $_.Exception.Message) }

# 2.6) league_status + injuries were built earlier as a fail-fast availability gate.

# 2.6a) Snapshot injuries counts (team-level + excluded players) for explainability caches
# NOTE: run after league_status so the snapshot can stay consistent with the player pool used downstream.
try {
  Write-Log ("Snapshot injuries counts cache for {0}" -f $Date)
  $injTool = Join-Path $RepoRoot 'tools/snapshot_injuries.py'
  if (Test-Path $injTool) {
    & $Python $injTool --date $Date 2>&1 | Tee-Object -FilePath $LogFile -Append | Out-Null
  } else {
    Write-Log 'snapshot_injuries.py missing; skipping injuries cache'
  }
} catch { Write-Log ("Injuries snapshot failed (non-fatal): {0}" -f $_.Exception.Message) }

# 3) Props predictions for today (calibrated) to CSV
# NOTE: --use-pure-onnx flag enables pure ONNX with NPU acceleration (NO sklearn required!)
# IMPORTANT: Restrict predictions to today's slate only (do NOT generate for all rostered players)
$SmartSimWorkers = $env:DAILY_SMARTSIM_WORKERS
if ($null -eq $SmartSimWorkers -or $SmartSimWorkers -eq '') { $SmartSimWorkers = $env:SMARTSIM_WORKERS }

# SmartSim knobs (shared with the later smart-sim-date step):
# - DAILY_SMARTSIM_NSIMS controls n_sims
# - DAILY_SMARTSIM_OVERWRITE controls overwrite behavior
# - DAILY_SKIP_SMARTSIM can skip SmartSim entirely (faster iteration)
$SmartSimNSims = $env:DAILY_SMARTSIM_NSIMS
if ($null -eq $SmartSimNSims -or $SmartSimNSims -eq '') {
  if ($IsCiRun) { $SmartSimNSims = '500' } else { $SmartSimNSims = '2000' }
}
try {
  $nTmp = [int]$SmartSimNSims
  if ($nTmp -lt 100) { $nTmp = 100 }
  if ($nTmp -gt 20000) { $nTmp = 20000 }
  $SmartSimNSims = [string]$nTmp
} catch {
  if ($IsCiRun) { $SmartSimNSims = '500' } else { $SmartSimNSims = '2000' }
}

$SmartSimOverwrite = $env:DAILY_SMARTSIM_OVERWRITE
$SkipSmartSim = $env:DAILY_SKIP_SMARTSIM
$SmartSimRosterMode = $env:DAILY_SMARTSIM_ROSTER_MODE
if ($null -eq $SmartSimRosterMode -or $SmartSimRosterMode -eq '') { $SmartSimRosterMode = 'pregame' }
$SmartSimOutPrefix = $env:DAILY_SMARTSIM_OUT_PREFIX
if ($null -eq $SmartSimOutPrefix -or $SmartSimOutPrefix -eq '') { $SmartSimOutPrefix = 'smart_sim' }

# Default SmartSim parallelism: if not explicitly configured, use a safe CPU-based default.
# This affects both predict-props SmartSim and the standalone smart-sim-date step.
try {
  if ($null -eq $SmartSimWorkers -or $SmartSimWorkers -eq '') {
    $cpu = [Environment]::ProcessorCount
    # Leave at 1 for very small machines; otherwise use (cpu-1) capped at 6.
    $auto = 1
    if ($cpu -ge 4) { $auto = [Math]::Min(6, [Math]::Max(2, $cpu - 1)) }
    $SmartSimWorkers = [string]$auto
    Write-Log ("SmartSim workers defaulted to {0} (set DAILY_SMARTSIM_WORKERS or SMARTSIM_WORKERS to override)" -f $SmartSimWorkers)
  }
} catch {}

$ppArgs = @(
  '-m','nba_betting.cli','predict-props',
  '--date', $Date,
  '--slate-only',
  '--calibrate','--calib-window','7',
  '--calibrate-player','--player-calib-window','30',
  '--use-pure-onnx',
  '--use-smart-sim',
  '--smart-sim-n-sims', $SmartSimNSims,
  '--smart-sim-pbp',
  '--smart-sim-roster-mode', $SmartSimRosterMode,
  '--smart-sim-out-prefix', $SmartSimOutPrefix
)

# Optional: disable SmartSim entirely for faster iteration
if ($null -ne $SkipSmartSim -and $SkipSmartSim -match '^(1|true|yes)$') {
  Write-Log 'Skipping SmartSim inside predict-props (DAILY_SKIP_SMARTSIM=1)'
  $ppArgs = @(
    '-m','nba_betting.cli','predict-props',
    '--date', $Date,
    '--slate-only',
    '--calibrate','--calib-window','7',
    '--calibrate-player','--player-calib-window','30',
    '--use-pure-onnx',
    '--no-use-smart-sim'
  )
} else {
  # Only overwrite SmartSim artifacts when explicitly requested.
  if ($null -ne $SmartSimOverwrite -and $SmartSimOverwrite -match '^(1|true|yes)$') {
    $ppArgs += '--smart-sim-overwrite'
  }
}
try {
  if ($null -ne $SkipSmartSim -and $SkipSmartSim -match '^(1|true|yes)$') {
    # no-op
  } elseif ($null -ne $SmartSimWorkers -and $SmartSimWorkers -match '^\d+$' -and [int]$SmartSimWorkers -gt 1) {
    Write-Log ("Using SmartSim parallel workers: {0}" -f $SmartSimWorkers)
    $ppArgs += @('--smart-sim-workers', $SmartSimWorkers)
  }
} catch {}

$rc3a = Invoke-PyMod -plist $ppArgs
Write-Log ("props-predictions exit code: {0}" -f $rc3a)
if ($rc3a -ne 0) {
  throw "predict-props failed with exit code $rc3a"
}
$propsPredictionsPath = Join-Path $RepoRoot ("data/processed/props_predictions_{0}.csv" -f $Date)
if (-not (Test-CsvHasDataRows -Path $propsPredictionsPath)) {
  throw "predict-props completed without writing data rows to $propsPredictionsPath"
}

# 3.0) Export SmartSim player quarter + scenario distributions for today's slate (non-fatal)
# Controlled by env DAILY_SKIP_SMARTSIM_PLAYER_SPLITS=1
try {
  $skipSplits = $env:DAILY_SKIP_SMARTSIM_PLAYER_SPLITS
  if ($null -ne $skipSplits -and $skipSplits -match '^(1|true|yes)$') {
    Write-Log 'Skipping SmartSim player splits export (DAILY_SKIP_SMARTSIM_PLAYER_SPLITS=1)'
  } else {
    $outQ = Join-Path $RepoRoot ("data/processed/smartsim_player_quarters_{0}.csv" -f $Date)
    $outS = Join-Path $RepoRoot ("data/processed/smartsim_player_scenarios_{0}.csv" -f $Date)
    Write-Log ("Exporting SmartSim player splits for {0}" -f $Date)
    $rcSplits = Invoke-PyMod -plist @(
      'tools/extract_smartsim_player_splits.py',
      '--start', $Date,
      '--end', $Date,
      '--out-quarters', $outQ,
      '--out-scenarios', $outS
    )
    Write-Log ("extract_smartsim_player_splits exit code: {0}" -f $rcSplits)
  }
} catch {
  Write-Log ("SmartSim player splits export failed (non-fatal): {0}" -f $_.Exception.Message)
}

# 3.1) Post-process props_predictions to drop OUT players (ensures downstream CSVs have no injured players at all)
try {
  $ppPath = Join-Path $RepoRoot ("data/processed/props_predictions_{0}.csv" -f $Date)
  $injExclPath = Join-Path $RepoRoot ("data/processed/injuries_excluded_{0}.csv" -f $Date)
  $leagueStatusPath = Join-Path $RepoRoot ("data/processed/league_status_{0}.csv" -f $Date)
  if (Test-Path $ppPath) {
  Write-Log "Filtering props_predictions to remove OUT players based on injuries_excluded list"

  # Repair injuries_excluded team assignments using processed rosters (prevents cross-team exclusions)
  try {
    $injRepair = Join-Path $RepoRoot 'tools/repair_injuries_excluded.py'
    if ((Test-Path $injRepair) -and (Test-Path $injExclPath)) {
      Write-Log "Repairing injuries_excluded team assignments via rosters"
      & $Python $injRepair --date $Date 2>&1 | Tee-Object -FilePath $LogFile -Append | Out-Null
    }
  } catch { Write-Log ("repair_injuries_excluded failed (non-fatal): {0}" -f $_.Exception.Message) }
  $filterScript = Join-Path $RepoRoot 'tools/filter_props_predictions_by_injuries.py'
  $outFilt = & $Python $filterScript --preds $ppPath --injuries $injExclPath --league-status $leagueStatusPath 2>&1 | Tee-Object -FilePath $LogFile -Append
  Write-Log ("Props predictions filter result: {0}" -f $outFilt)
  } else {
  Write-Log "No props_predictions file found to filter; skipping"
  }
} catch { Write-Log ("Props predictions filter failed (non-fatal): {0}" -f $_.Exception.Message) }

# 3.2) Build pregame expected minutes for today (leakage-safe: rotations history up to yesterday)
# Controlled by env DAILY_SKIP_PREGAME_EXPECTED_MINUTES=1
try {
  $skipPem = $env:DAILY_SKIP_PREGAME_EXPECTED_MINUTES
  if ($null -eq $skipPem -or $skipPem -notmatch '^(1|true|yes)$') {
    $pemLookback = $env:DAILY_PREGAME_EXPECTED_MINUTES_LOOKBACK_DAYS
    if ($null -eq $pemLookback -or $pemLookback -eq '') { $pemLookback = '60' }
    $pemHalfLife = $env:DAILY_PREGAME_EXPECTED_MINUTES_HALF_LIFE_DAYS
    if ($null -eq $pemHalfLife -or $pemHalfLife -eq '') { $pemHalfLife = '12' }
    $pemAlpha = $env:DAILY_PREGAME_EXPECTED_MINUTES_BLEND_ALPHA
    if ($null -eq $pemAlpha -or $pemAlpha -eq '') { $pemAlpha = '1.0' }

    # Sanity guard: only use rotations-derived expected minutes if rotations history is present and fresh through yesterday.
    $pemSource = 'rotations'
    try {
      $rotHist = Join-Path $RepoRoot 'data/processed/rotation_stints_history.csv'
      if (-not (Test-Path $rotHist)) {
        $pemSource = 'props'
        Write-Log 'pregame_expected_minutes: rotation_stints_history.csv missing; falling back to props-based minutes'
      } else {
        $env:ROT_HIST_PATH = $rotHist
        $env:ROT_HIST_YESTERDAY = $yesterday
        $pyRotOk = @'
import os
from pathlib import Path

import pandas as pd

fp = Path(os.environ.get("ROT_HIST_PATH", ""))
y = (os.environ.get("ROT_HIST_YESTERDAY", "") or "").strip()
try:
    df = pd.read_csv(fp, usecols=["date"])
except Exception:
    print("0")
    raise SystemExit(0)

if df is None or df.empty or "date" not in df.columns:
    print("0")
    raise SystemExit(0)

d = pd.to_datetime(df["date"], errors="coerce").dropna()
if d.empty:
    print("0")
    raise SystemExit(0)

mx = d.max().strftime("%Y-%m-%d")
print("1" if (mx >= y and y) else "0")
'@
        $tmpPyPemRot = Join-Path $LogPath ("pem_rotations_fresh_{0}.py" -f $Stamp)
        Set-Content -Path $tmpPyPemRot -Value $pyRotOk -Encoding UTF8
        $rotOk = (& $Python $tmpPyPemRot 2>$null | Select-Object -First 1)
        if ($null -eq $rotOk -or [string]$rotOk -notmatch '^(1)$') {
          $pemSource = 'props'
          Write-Log ("pregame_expected_minutes: rotations history stale (need >= {0}); falling back to props-based minutes" -f $yesterday)
        }
      }
    } catch {
      $pemSource = 'props'
      Write-Log ("pregame_expected_minutes: rotations freshness check failed; falling back to props-based minutes ({0})" -f $_.Exception.Message)
    }

    if ($pemSource -eq 'rotations') {
      Write-Log ("Building pregame_expected_minutes for {0} from ESPN rotations history (lookback={1}d, half-life={2}d, alpha={3})" -f $Date, $pemLookback, $pemHalfLife, $pemAlpha)
      $rcPem = Invoke-PyMod -plist @(
        'tools/build_pregame_expected_minutes_range.py',
        '--start', $Date,
        '--end', $Date,
        '--source', 'rotations',
        '--rotations-lookback-days', $pemLookback,
        '--rotations-half-life-days', $pemHalfLife,
        '--rotations-blend-alpha', $pemAlpha,
        '--overwrite'
      )
    } else {
      Write-Log ("Building pregame_expected_minutes for {0} from props roll-minutes (fallback)" -f $Date)
      $rcPem = Invoke-PyMod -plist @(
        'tools/build_pregame_expected_minutes_range.py',
        '--start', $Date,
        '--end', $Date,
        '--source', 'props',
        '--overwrite'
      )
    }
    Write-Log ("build_pregame_expected_minutes exit code: {0}" -f $rcPem)
  } else {
    Write-Log 'Skipping pregame_expected_minutes build (DAILY_SKIP_PREGAME_EXPECTED_MINUTES=1)'
  }
} catch { Write-Log ("pregame_expected_minutes build failed (non-fatal): {0}" -f $_.Exception.Message) }

# 4) Props actuals snapshot safeguard for yesterday (CLI fetch already ran earlier)
try {
  $snapPath = Join-Path $RepoRoot ("data/processed/props_actuals_{0}.csv" -f $yesterday)
  if (-not (Test-Path $snapPath)) {
    Write-Log "Snapshot $($snapPath) missing; attempting to derive from parquet store"
    $parq = Join-Path $RepoRoot 'data/processed/props_actuals.parquet'
    if (Test-Path $parq) {
      try {
        # Use python to extract rows for that date (one-liner to avoid here-string parsing issues)
        $pycode = (
          "import pandas as pd; parq=r'" + $parq + "'; date='" + $yesterday + "'; out=r'" + $snapPath + "'; " +
          "df=pd.read_parquet(parq); " +
          "\nimport pandas as pd; " +
          "\nimport numpy as np; " +
          "\nimport sys; " +
          "\n" +
          "\n" +
          "\nif not df.empty:\n" +
          "    df['date']=pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d');\n" +
          "    day=df[df['date']==date];\n" +
          "    (day.to_csv(out, index=False) if not day.empty else None)"
        )
        $null = & $Python -c $pycode
        if (Test-Path $snapPath) { Write-Log "Derived missing snapshot for $yesterday" } else { Write-Log "No rows found in parquet for $yesterday; snapshot not created" }
      } catch { Write-Log ("Snapshot derive failed: {0}" -f $_.Exception.Message) }
    } else {
      Write-Log "Parquet store missing; cannot derive snapshot"
    }
  }
} catch { Write-Log ("Snapshot safeguard error: {0}" -f $_.Exception.Message) }

# 5) Props lines + edges + recommendations for today via the shared OddsAPI refresh worker.
$edgesPath = Join-Path $RepoRoot ("data/processed/props_edges_{0}.csv" -f $Date)
$propsRecsPath = Join-Path $RepoRoot ("data/processed/props_recommendations_{0}.csv" -f $Date)

# Optional: control runtime probability calibration strength in props edge scoring.
# - Set DAILY_PROPS_PROB_CALIB_ALPHA to override for this run.
# - Or set PROPS_PROB_CALIB_ALPHA directly in the environment.
try {
  $dailyAlpha = $env:DAILY_PROPS_PROB_CALIB_ALPHA
  if ($null -ne $dailyAlpha -and $dailyAlpha.Trim() -ne '') {
    try {
      $a = [double]$dailyAlpha
      if ($a -lt 0) { $a = 0 }
      if ($a -gt 1) { $a = 1 }
      $env:PROPS_PROB_CALIB_ALPHA = ([string]$a)
      Write-Log ("Using PROPS_PROB_CALIB_ALPHA={0} (from DAILY_PROPS_PROB_CALIB_ALPHA)" -f $a)
    } catch {
      Write-Log ("Invalid DAILY_PROPS_PROB_CALIB_ALPHA='{0}' (ignoring)" -f $dailyAlpha)
    }
  } elseif ($null -ne $env:PROPS_PROB_CALIB_ALPHA -and $env:PROPS_PROB_CALIB_ALPHA.Trim() -ne '') {
    Write-Log ("Using PROPS_PROB_CALIB_ALPHA={0}" -f $env:PROPS_PROB_CALIB_ALPHA)
  }
} catch { Write-Log ("PROPS_PROB_CALIB_ALPHA wiring failed (non-fatal): {0}" -f $_.Exception.Message) }

$rc4a = Invoke-SharedPropsRefreshWorker -TargetDate $Date
Write-Log ("shared props refresh exit code: {0}" -f $rc4a)
if ($rc4a -ne 0) {
  throw "shared props refresh failed with exit code $rc4a"
}

# 6) Export recommendations CSVs for site consumption
# 6a) Game recommendations from predictions + odds
$maxPlusOdds = $env:DAILY_MAX_PLUS_ODDS
$exportGamesArgs = @('-m','nba_betting.cli','export-recommendations','--date', $Date)
if ($null -ne $maxPlusOdds -and $maxPlusOdds -ne '') {
  try {
    $mpo = [double]$maxPlusOdds
    $exportGamesArgs += @('--max-plus-odds', ([string]$mpo))
    if ($mpo -gt 0) {
      Write-Log ("Applying odds guard to game exports: max_plus_odds={0}" -f $mpo)
    } else {
      Write-Log ("Disabling odds guard for game exports: max_plus_odds={0}" -f $mpo)
    }
  } catch {
    Write-Log ("Invalid DAILY_MAX_PLUS_ODDS='{0}' (skipping odds guard)" -f $maxPlusOdds)
  }
}
$rc5 = Invoke-PyMod -plist $exportGamesArgs
Write-Log ("export-recommendations exit code: {0}" -f $rc5)
# 6b) High-confidence picks (blended scoring) for games
try {
  Write-Log ("Generating high-confidence picks for {0}" -f $Date)
  $rc5b = Invoke-PyMod -plist @(
    '-m','nba_betting.cli','recommend-picks',
    '--date', $Date,
    '--topN','10',
    '--minScore','0.15',
    '--minAtsEdge','0.05',
    '--minAtsEV','0.00',
    '--atsBlend','0.25',
    '--minTotalEdge','0.02',
    '--minTotalEV','0.00',
    '--totalsBlend','0.10'
  )
  Write-Log ("recommend-picks exit code: {0}" -f $rc5b)
} catch {
  Write-Log ("recommend-picks failed (non-fatal): {0}" -f $_.Exception.Message)
}
# 6c) Props recommendations are generated by the shared OddsAPI props refresh worker above.

# 6c.0) Optional: export per-game Top-N props JSON (non-fatal)
# Uses the Flask endpoint logic in-process (no server required).
# Controlled by env DAILY_SKIP_PROPS_TOP_BY_GAME=1
try {
  $skipTopByGame = $env:DAILY_SKIP_PROPS_TOP_BY_GAME
  if ($null -ne $skipTopByGame -and $skipTopByGame -match '^(1|true|yes)$') {
    Write-Log 'Skipping props per-game Top-N export (DAILY_SKIP_PROPS_TOP_BY_GAME=1)'
  } else {
    $perGameLimit = $env:DAILY_PROPS_PER_GAME_LIMIT
    if ($null -eq $perGameLimit -or $perGameLimit -eq '') { $perGameLimit = '3' }
    try { $perGameLimitInt = [int]$perGameLimit } catch { $perGameLimitInt = 3 }
    if ($perGameLimitInt -lt 1) { $perGameLimitInt = 1 }
    if ($perGameLimitInt -gt 10) { $perGameLimitInt = 10 }

    $slateLimit = $env:DAILY_PROPS_SLATE_LIMIT
    if ($null -eq $slateLimit -or $slateLimit -eq '') { $slateLimit = '25' }
    try { $slateLimitInt = [int]$slateLimit } catch { $slateLimitInt = 25 }
    if ($slateLimitInt -lt 1) { $slateLimitInt = 1 }
    if ($slateLimitInt -gt 200) { $slateLimitInt = 200 }

    $slatePerMarketLimit = $env:DAILY_PROPS_SLATE_PER_MARKET_LIMIT
    if ($null -eq $slatePerMarketLimit -or $slatePerMarketLimit -eq '') { $slatePerMarketLimit = '4' }
    try { $slatePerMarketLimitInt = [int]$slatePerMarketLimit } catch { $slatePerMarketLimitInt = 4 }
    if ($slatePerMarketLimitInt -lt 1) { $slatePerMarketLimitInt = 1 }
    if ($slatePerMarketLimitInt -gt 50) { $slatePerMarketLimitInt = 50 }

    $mkts = $env:DAILY_PROPS_MARKETS
    if ($null -eq $mkts -or $mkts -eq '') { $mkts = 'pts,reb,ast,threes,blk,stl,pra,pr,pa,ra,dd,td' }

    $outTopByGame = Join-Path $RepoRoot ("data/processed/props_recommendations_top_by_game_{0}.json" -f $Date)
    Write-Log ("Exporting props per-game + per-market Top-N JSON for {0} (per_game_limit={1}, slate_per_market_limit={2}, slate_limit={3}, markets={4})" -f $Date, $perGameLimitInt, $slatePerMarketLimitInt, $slateLimitInt, $mkts)

    $tmpPyTop = Join-Path $LogPath ("export_props_top_by_game_{0}.py" -f $Stamp)
    $pyTop = @"
import json
import sys
from pathlib import Path

repo_root = Path(r"{REPO_PLACEHOLDER}")
if str(repo_root) not in sys.path:
  sys.path.insert(0, str(repo_root))

import app

date_str = r"{DATE_PLACEHOLDER}"
out_path = Path(r"{OUT_PLACEHOLDER}")
per_game_limit = int(r"{PGL_PLACEHOLDER}")
slate_limit = int(r"{SL_PLACEHOLDER}")
markets = r"{MKTS_PLACEHOLDER}".strip()
slate_per_market_limit = int(r"{SPML_PLACEHOLDER}")

q = f"/api/props/recommendations?date={date_str}&compact=1&portfolio_only=1&use_snapshot=0&limit={slate_limit}&per_game_limit={per_game_limit}&per_market=1&slate_per_market_limit={slate_per_market_limit}"
if markets:
  q += "&markets=" + markets

client = app.app.test_client()
resp = client.get(q)
try:
  payload = resp.get_json() if resp is not None else None
except Exception:
  payload = None

if not isinstance(payload, dict):
  payload = {"error": "no_json", "status": int(getattr(resp, 'status_code', 0) or 0)}

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print("OK")
"@
    $pyTop = $pyTop.Replace('{DATE_PLACEHOLDER}', $Date)
  $pyTop = $pyTop.Replace('{REPO_PLACEHOLDER}', $RepoRoot)
    $pyTop = $pyTop.Replace('{OUT_PLACEHOLDER}', $outTopByGame)
    $pyTop = $pyTop.Replace('{PGL_PLACEHOLDER}', $perGameLimitInt)
    $pyTop = $pyTop.Replace('{SL_PLACEHOLDER}', $slateLimitInt)
    $pyTop = $pyTop.Replace('{MKTS_PLACEHOLDER}', $mkts)
    $pyTop = $pyTop.Replace('{SPML_PLACEHOLDER}', $slatePerMarketLimitInt)
    Set-Content -Path $tmpPyTop -Value $pyTop -Encoding UTF8
    $outTop = & $Python $tmpPyTop 2>&1 | Tee-Object -FilePath $LogFile -Append
    if ($outTop -match 'OK') { Write-Log ("Wrote {0}" -f $outTopByGame) } else { Write-Log ("Top-by-game export returned: {0}" -f $outTop) }
  }
} catch {
  Write-Log ("Props per-game Top-N export failed (non-fatal): {0}" -f $_.Exception.Message)
}

# 6d) Export authoritative best-edges snapshots (games + props) for tracking/UI
try {
  Write-Log ("Exporting best-edges snapshots for {0}" -f $Date)
  $rc6d = Invoke-PyMod -plist @('-m','nba_betting.cli','export-best-edges','--date', $Date, '--overwrite')
  Write-Log ("export-best-edges exit code: {0}" -f $rc6d)
} catch {
  Write-Log ("export-best-edges failed (non-fatal): {0}" -f $_.Exception.Message)
}

# 6e) Purge cached summary so endpoints cannot serve stale results
try {
  $sumCache = Join-Path $RepoRoot 'data/processed/recommendations_summary.json'
  if (Test-Path $sumCache) {
    Remove-Item $sumCache -Force -ErrorAction SilentlyContinue
    Write-Log 'Purged recommendations_summary.json cache'
  } else {
    Write-Log 'No recommendations_summary.json cache to purge'
  }
} catch {
  Write-Log ("Summary cache purge failed (non-fatal): {0}" -f $_.Exception.Message)
}

# 6c.1) Post-process props_recommendations to prefer regular-priced plays and add explainability
try {
  Write-Log ("Post-processing props_recommendations for {0}: prefer regular-priced plays, add reasons" -f $Date)
  $propsCsv = Join-Path $RepoRoot ("data/processed/props_recommendations_{0}.csv" -f $Date)
  if (Test-Path $propsCsv) {
  $tmpPy3 = Join-Path $LogPath ("props_recs_regular_patch_{0}.py" -f $Stamp)
  $pycode3 = @'
import json, ast, math
import pandas as pd
from pathlib import Path

date_str = "{DATE_PLACEHOLDER}"
repo_root = Path(r"{REPO_PLACEHOLDER}")
props_csv = repo_root / f"data/processed/props_recommendations_{date_str}.csv"
preds_csv = repo_root / f"data/processed/props_predictions_{date_str}.csv"

def _parse_obj(val):
  if isinstance(val, (list, dict)):
    return val
  s = str(val)
  if not s or s.strip() in {"", "None", "nan"}:
    return None
  try:
    return json.loads(s)
  except Exception:
    try:
      return ast.literal_eval(s)
    except Exception:
      return None

def _regular_price(pr):
  try:
    if pr is None or (isinstance(pr, float) and math.isnan(pr)):
      return False
    v = float(pr)
    return (-150.0 <= v <= 150.0)
  except Exception:
    return False

def _choose_top_play(plays):
  if not isinstance(plays, list) or not plays:
    return None
  # Prefer regular-priced plays, fallback to any
  regular = [p for p in plays if _regular_price(p.get("price"))]
  cand = regular if regular else plays

  def eligible(p):
    mkt = str(p.get("market") or "").lower()
    if mkt in {"pts", "pra"}:
      edge = p.get("edge")
      try:
        return edge is not None and abs(float(edge)) >= 0.15
      except Exception:
        return False
    return True

  # Prefer core markets when available (avoid pts/pra unless very strong)
  core = {"reb", "ra", "ast"}
  core_cand = [p for p in cand if str(p.get("market") or "").lower() in core and eligible(p)]
  elig = [p for p in cand if eligible(p)]
  cand = core_cand if core_cand else (elig if elig else cand)

  def score(p):
    evp = p.get("ev_pct")
    if evp is not None:
      try:
        return float(evp)
      except Exception:
        pass
    ev = p.get("ev")
    try:
      return float(ev) * 100.0 if ev is not None else 0.0
    except Exception:
      return 0.0
  cand = sorted(cand, key=score, reverse=True)
  p = cand[0]
  return {
    "market": p.get("market"),
    "side": p.get("side"),
    "line": p.get("line"),
    "price": p.get("price"),
    "ev": p.get("ev"),
    "ev_pct": p.get("ev_pct"),
    "book": p.get("book"),
  }

def _build_model_map(preds_df):
  m = {}
  if isinstance(preds_df, pd.DataFrame) and not preds_df.empty:
    tmp = preds_df.copy()
    for c in ("player_name","team"):
      if c not in tmp.columns:
        tmp[c] = None
    def _stat_map(row):
      out = {}
      for col, key in [("pred_pts","pts"),("pred_reb","reb"),("pred_ast","ast"),("pred_threes","threes"),("pred_pra","pra")]:
        if col in tmp.columns:
          try:
            v = float(row.get(col))
            if not math.isnan(v):
              out[key] = v
          except Exception:
            pass
      return out
    tmp["_stat_map"] = tmp.apply(_stat_map, axis=1)
    for _, r in tmp.iterrows():
      k = (str(r.get("player_name") or "").strip().lower(), str(r.get("team") or "").strip().upper())
      m[k] = r.get("_stat_map") or {}
  return m

def _explain_baseline(row, model_map):
  tp = row.get("top_play")
  if not isinstance(tp, dict) or not tp:
    return ""
  mkt = str(tp.get("market") or "").lower()
  line = tp.get("line")
  player = str(row.get("player") or row.get("player_name") or "").strip().lower()
  team = str(row.get("team") or "").strip().upper()
  stats = model_map.get((player, team)) or {}
  base = stats.get(mkt)
  if base is not None and line is not None:
    try:
      delta = float(base) - float(line)
      sign = "+" if delta >= 0 else ""
      return f"model {float(base):.1f} vs line {float(line):.1f} ({sign}{float(delta):.1f})"
    except Exception:
      return ""
  return ""

def _consensus_line_adv(row):
  tp = row.get("top_play") or {}
  plays = row.get("_plays_list") or []
  reasons = []
  cons_norm = 0.0
  line_adv_norm = 0.0
  # EV reason
  evp = tp.get("ev_pct")
  ev = tp.get("ev")
  if evp is not None:
    try:
      reasons.append(f"EV {float(evp):.1f}%")
    except Exception:
      pass
  elif ev is not None:
    try:
      reasons.append(f"EV +{float(ev):.2f}")
    except Exception:
      pass
  # Price friendliness and regular tag
  pr = tp.get("price")
  if pr is not None:
    try:
      if abs(float(pr) + 110.0) <= 10.0:
        reasons.append("Friendly price (~-110)")
      if _regular_price(pr):
        reasons.append("Regular price range (-150 to +150)")
    except Exception:
      pass
  # Consensus: same market/side across books (prefer regular-priced set)
  mk = str(tp.get("market") or "").lower()
  side = str(tp.get("side") or "").upper()
  same_all = [p for p in (plays or []) if str(p.get("market") or "").lower() == mk and str(p.get("side") or "").upper() == side]
  same_regular = [p for p in same_all if _regular_price(p.get("price"))]
  same = same_regular if same_regular else same_all
  distinct_books = sorted(list({str(p.get("book") or "").lower() for p in same if p.get("book") is not None}))
  n_books = len(distinct_books)
  if n_books >= 3:
    reasons.append(f"Consensus: {n_books} books aligned")
  cons_norm = max(0.0, min(1.0, (n_books - 1) / 4.0))
  # Line advantage
  try:
    lines = [float(p.get("line")) for p in same if p.get("line") is not None]
    tpl = float(tp.get("line")) if tp.get("line") is not None else None
    if lines and tpl is not None:
      if side == "OVER":
        best = min(lines)
        if tpl <= best + 1e-6:
          reasons.append("Best line available")
          line_adv_norm = 1.0
      elif side == "UNDER":
        best = max(lines)
        if tpl >= best - 1e-6:
          reasons.append("Best line available")
          line_adv_norm = 1.0
  except Exception:
    pass
  return reasons, cons_norm, line_adv_norm

df = pd.read_csv(props_csv)
preds_df = pd.read_csv(preds_csv) if preds_csv.exists() else pd.DataFrame()
model_map = _build_model_map(preds_df)

# Parse plays and compute top_play
df = df.copy()
df["_plays_list"] = df.apply(lambda r: _parse_obj(r.get("plays")), axis=1)
df["top_play"] = df["_plays_list"].map(_choose_top_play)

# Explain baseline
df["top_play_explain"] = df.apply(lambda r: _explain_baseline(r, model_map), axis=1)

# Baseline raw value for scoring
def _baseline_val(row):
  tp = row.get("top_play")
  if not isinstance(tp, dict) or not tp:
    return None
  player = str(row.get("player") or row.get("player_name") or "").strip().lower()
  team = str(row.get("team") or "").strip().upper()
  stats = model_map.get((player, team)) or {}
  m = str(tp.get("market") or "").lower()
  return stats.get(m)
df["top_play_baseline"] = df.apply(_baseline_val, axis=1)

# Consensus, line-advantage, reasons
res = df.apply(lambda r: pd.Series({
  "_reasons_cons_line": _consensus_line_adv(r)
}), axis=1)
df["top_play_reasons"] = res["_reasons_cons_line"].map(lambda x: (x[0] if isinstance(x, tuple) else []))
df["top_play_consensus"] = res["_reasons_cons_line"].map(lambda x: (x[1] if isinstance(x, tuple) else 0.0))
df["top_play_line_adv"] = res["_reasons_cons_line"].map(lambda x: (x[2] if isinstance(x, tuple) else 0.0))
df.drop(columns=["_reasons_cons_line"], inplace=True, errors="ignore")

# Write back with enriched columns; preserve existing columns
df.to_csv(props_csv, index=False)
print("OK")
'@
  $pycode3 = $pycode3.Replace('{DATE_PLACEHOLDER}', $Date).Replace('{REPO_PLACEHOLDER}', $RepoRoot)
  Set-Content -Path $tmpPy3 -Value $pycode3 -Encoding UTF8
  $out3 = & $Python $tmpPy3 2>&1 | Tee-Object -FilePath $LogFile -Append
  if ($out3 -match 'OK') { Write-Log 'Props recommendations patched with regular-priced preference and reasons' } else { Write-Log ("Props recommendations patch returned: {0}" -f $out3) }
  } else {
  Write-Log 'No props_recommendations CSV found; skipping post-process'
  }
} catch {
  Write-Log ("Props recommendations post-process failed (non-fatal): {0}" -f $_.Exception.Message)
}

# 6c.1.5) Build curated slate JSON artifact for /recommendations (non-fatal)
# Uses the Flask endpoint logic in-process (no server required).
# Controlled by env DAILY_SKIP_SLATE_JSON=1
try {
  $skipSlate = $env:DAILY_SKIP_SLATE_JSON
  if ($null -ne $skipSlate -and $skipSlate -match '^(1|true|yes)$') {
    Write-Log 'Skipping curated slate JSON build (DAILY_SKIP_SLATE_JSON=1)'
  } else {
    $defaultSlateQuery = ("/recommendations?format=json&view=slate&date={0}" -f $Date)
    $slateQuery = $env:DAILY_SLATE_QUERY
    $slateQueryExtra = $env:DAILY_SLATE_QUERY_EXTRA
    if ($null -eq $slateQuery -or $slateQuery -eq '') {
      $slateQuery = $defaultSlateQuery
      if ($null -ne $slateQueryExtra -and $slateQueryExtra -ne '') {
        if ($slateQueryExtra.StartsWith('&') -or $slateQueryExtra.StartsWith('?')) {
          $slateQuery = ($slateQuery + $slateQueryExtra)
        } else {
          $slateQuery = ($slateQuery + '&' + $slateQueryExtra)
        }
      }
    }

    $slateOutEnv = $env:DAILY_SLATE_OUT
    $forceDefaultOut = $env:DAILY_SLATE_FORCE_DEFAULT_OUT
    if ($null -ne $slateOutEnv -and $slateOutEnv -ne '') {
      $slateOut = $slateOutEnv
    } else {
      if ($slateQuery -ne $defaultSlateQuery -and -not ($null -ne $forceDefaultOut -and $forceDefaultOut -match '^(1|true|yes)$')) {
        $slateOut = Join-Path $RepoRoot ("data/processed/recommendations_slate_{0}_custom.json" -f $Date)
      } else {
        $slateOut = Join-Path $RepoRoot ("data/processed/recommendations_slate_{0}.json" -f $Date)
      }
    }
    try { if (Test-Path $slateOut) { Remove-Item $slateOut -Force -ErrorAction SilentlyContinue } } catch { }
    Write-Log ("Building curated slate JSON for {0} -> {1}" -f $Date, $slateOut)

    $tmpPySlate = Join-Path $LogPath ("build_recommendations_slate_{0}.py" -f $Stamp)
    $pySlate = @"
import json
import sys
from pathlib import Path

repo_root = Path(r"{REPO_PLACEHOLDER}")
if str(repo_root) not in sys.path:
  sys.path.insert(0, str(repo_root))

import app

date_str = r"{DATE_PLACEHOLDER}"
out_path = Path(r"{OUT_PLACEHOLDER}")

q = r"{QUERY_PLACEHOLDER}"

client = app.app.test_client()
resp = client.get(q)
try:
  payload = resp.get_json() if resp is not None else None
except Exception:
  payload = None

if not isinstance(payload, dict):
  payload = {"error": "no_json", "status": int(getattr(resp, 'status_code', 0) or 0)}

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print("OK")
"@
    $pySlate = $pySlate.Replace('{DATE_PLACEHOLDER}', $Date)
  $pySlate = $pySlate.Replace('{REPO_PLACEHOLDER}', $RepoRoot)
    $pySlate = $pySlate.Replace('{OUT_PLACEHOLDER}', $slateOut)
    $pySlate = $pySlate.Replace('{QUERY_PLACEHOLDER}', $slateQuery)
    Set-Content -Path $tmpPySlate -Value $pySlate -Encoding UTF8
    $outSlate = & $Python $tmpPySlate 2>&1 | Tee-Object -FilePath $LogFile -Append
    if ($outSlate -match 'OK') { Write-Log ("Wrote {0}" -f $slateOut) } else { Write-Log ("Curated slate build returned: {0}" -f $outSlate) }
  }
} catch {
  Write-Log ("Curated slate JSON build failed (non-fatal): {0}" -f $_.Exception.Message)
}

# 6c.1.6) Build cards props snapshot artifact for /api/cards (non-fatal)
try {
  $skipCardsPropsSnapshot = $env:DAILY_SKIP_CARDS_PROPS_SNAPSHOT
  if ($null -ne $skipCardsPropsSnapshot -and $skipCardsPropsSnapshot -match '^(1|true|yes)$') {
    Write-Log 'Skipping cards props snapshot build (DAILY_SKIP_CARDS_PROPS_SNAPSHOT=1)'
  } else {
    $cardsPropsSource = [string]$env:DAILY_CARDS_PROPS_SNAPSHOT_SOURCE
    if ([string]::IsNullOrWhiteSpace($cardsPropsSource)) { $cardsPropsSource = 'source' }
    $cardsPropsSource = $cardsPropsSource.Trim().ToLowerInvariant()
    if (@('auto', 'source', 'runtime', 'snapshot') -notcontains $cardsPropsSource) { $cardsPropsSource = 'auto' }
    $cardsPropsOut = Join-Path $RepoRoot ("data/processed/cards_props_snapshot_{0}.json" -f $Date)
    try { if (Test-Path $cardsPropsOut) { Remove-Item $cardsPropsOut -Force -ErrorAction SilentlyContinue } } catch { }
    Write-Log ("Building cards props snapshot for {0} -> {1} (props_source={2})" -f $Date, $cardsPropsOut, $cardsPropsSource)

    $tmpPyCardsProps = Join-Path $LogPath ("build_cards_props_snapshot_{0}.py" -f $Stamp)
    $pyCardsProps = @"
import json
import sys
from pathlib import Path

repo_root = Path(r"{REPO_PLACEHOLDER}")
if str(repo_root) not in sys.path:
  sys.path.insert(0, str(repo_root))

import app

date_str = r"{DATE_PLACEHOLDER}"
out_path = Path(r"{OUT_PLACEHOLDER}")
props_source = r"{PROPS_SOURCE_PLACEHOLDER}"

client = app.app.test_client()
resp = client.get(f"/api/cards?date={date_str}&props_source={props_source}")
try:
  payload = resp.get_json() if resp is not None else None
except Exception:
  payload = None

games_out = []
if isinstance(payload, dict):
  for game in (payload.get("games") or []):
    if not isinstance(game, dict):
      continue
    prop_recommendations = game.get("prop_recommendations") if isinstance(game.get("prop_recommendations"), dict) else {}
    games_out.append({
      "home_tri": game.get("home_tri"),
      "away_tri": game.get("away_tri"),
      "prop_recommendations": {
        "home": [row for row in (prop_recommendations.get("home") or []) if isinstance(row, dict)],
        "away": [row for row in (prop_recommendations.get("away") or []) if isinstance(row, dict)],
      },
    })

out = {
  "date": date_str,
  "games": games_out,
}
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
print("OK")
"@
    $pyCardsProps = $pyCardsProps.Replace('{DATE_PLACEHOLDER}', $Date)
    $pyCardsProps = $pyCardsProps.Replace('{REPO_PLACEHOLDER}', $RepoRoot)
    $pyCardsProps = $pyCardsProps.Replace('{OUT_PLACEHOLDER}', $cardsPropsOut)
    $pyCardsProps = $pyCardsProps.Replace('{PROPS_SOURCE_PLACEHOLDER}', $cardsPropsSource)
    Set-Content -Path $tmpPyCardsProps -Value $pyCardsProps -Encoding UTF8
    $outCardsProps = & $Python $tmpPyCardsProps 2>&1 | Tee-Object -FilePath $LogFile -Append
    if ($outCardsProps -match 'OK') { Write-Log ("Wrote {0}" -f $cardsPropsOut) } else { Write-Log ("Cards props snapshot build returned: {0}" -f $outCardsProps) }
  }
} catch {
  Write-Log ("Cards props snapshot build failed (non-fatal): {0}" -f $_.Exception.Message)
}

# 6c.2) Props reliability (60d) and probability calibration JSON (local-only)
try {
  Write-Log "Computing props reliability bins (60d)"
  $relScript = Join-Path $RepoRoot 'tools/compute_props_reliability.py'
  if (Test-Path $relScript) {
    & $Python $relScript 2>&1 | Tee-Object -FilePath $LogFile -Append | Out-Null
    $calScript = Join-Path $RepoRoot 'tools/calibrate_props_probability.py'
    if (Test-Path $calScript) {
      Write-Log "Generating props probability calibration JSON"
      & $Python $calScript 2>&1 | Tee-Object -FilePath $LogFile -Append | Out-Null
    } else { Write-Log 'Calibration script missing; skipping props_prob_calibration.json' }
  } else { Write-Log 'Props reliability script missing; skipping' }
} catch { Write-Log ("Props reliability/calibration block failed (non-fatal): {0}" -f $_.Exception.Message) }

# 7) PBP-derived markets for today's slate (tip winner, first basket, early threes)
try {
  Write-Log ("Predicting PBP-derived markets for {0}" -f $Date)
  $rcPbp = Invoke-PyMod -plist @('-m','nba_betting.cli','predict-pbp-markets','--date', $Date)
  Write-Log ("predict-pbp-markets exit code: {0}" -f $rcPbp)
} catch {
  Write-Log ("predict-pbp-markets failed (non-fatal): {0}" -f $_.Exception.Message)
}

# 7.1a) First-basket recommendations for today's slate
try {
  Write-Log ("Exporting first-basket recommendations for {0}" -f $Date)
  $rcFbRecs = Invoke-PyMod -plist @('-m','nba_betting.cli','first-basket-recs','--date', $Date)
  Write-Log ("first-basket-recs exit code: {0}" -f $rcFbRecs)
} catch {
  Write-Log ("first-basket-recs failed (non-fatal): {0}" -f $_.Exception.Message)
}

# 7.1) Export compact game cards for frontend
try {
  Write-Log ("Exporting game cards for {0}" -f $Date)
  $rcCards = Invoke-PyMod -plist @('-m','nba_betting.cli','export-game-cards','--date', $Date)
  Write-Log ("export-game-cards exit code: {0}" -f $rcCards)
} catch {
  Write-Log ("export-game-cards failed (non-fatal): {0}" -f $_.Exception.Message)
}

# 7.15) Optional refinement: build interval band calibration (p10/p90 widening) from recent finals.
# Runs BEFORE SmartSim so today's interval ladders can use the latest calibration.
try {
  $doIntervalBandCalib = $env:DAILY_INTERVAL_BAND_CALIB
  if ($null -eq $doIntervalBandCalib -or $doIntervalBandCalib -eq '') { $doIntervalBandCalib = '0' }
  if ($doIntervalBandCalib -match '^(1|true|yes)$') {
    $endEval = (Get-Date $Date).AddDays(-1).ToString('yyyy-MM-dd')
    $startEval = (Get-Date $Date).AddDays(-7).ToString('yyyy-MM-dd')
    Write-Log ("Building interval actuals + evaluation + band calibration (start={0}, end={1})" -f $startEval, $endEval)

    $rcAct = Invoke-PyMod -plist @('tools/build_interval_actuals_from_pbp_espn.py','--start', $startEval, '--end', $endEval)
    Write-Log ("build_interval_actuals_from_pbp_espn exit code: {0}" -f $rcAct)

    $rcEval = Invoke-PyMod -plist @('tools/evaluate_intervals.py','--start', $startEval, '--end', $endEval, '--use-pbp-only')
    Write-Log ("evaluate_intervals exit code: {0}" -f $rcEval)

    $rcCal = Invoke-PyMod -plist @('tools/build_intervals_band_calibration.py','--start', $startEval, '--end', $endEval)
    Write-Log ("build_intervals_band_calibration exit code: {0}" -f $rcCal)
  } else {
    Write-Log 'Skipping interval band calibration (set DAILY_INTERVAL_BAND_CALIB=1 to enable)'
  }
} catch {
  Write-Log ("Interval band calibration step failed (non-fatal): {0}" -f $_.Exception.Message)
}

# 7.2) SmartSim distributions for today's slate (writes per-game smart_sim_<date>_<HOME>_<AWAY>.json)
try {
  $skipSmart = $env:DAILY_SKIP_SMARTSIM
  if ($null -eq $skipSmart -or $skipSmart -notmatch '^(1|true|yes)$') {
    # If predict-props already generated today's smart_sim_<date>_*.json artifacts,
    # avoid re-running the full SmartSim slate (this is the single biggest runtime sink).
    $forceSmart = $env:DAILY_FORCE_SMARTSIM_DATE
    if ($null -eq $forceSmart -or $forceSmart -eq '') { $forceSmart = '0' }
    $existingSmart = @(Get-ChildItem (Join-Path $RepoRoot ("data/processed/{0}_{1}_*.json" -f $SmartSimOutPrefix, $Date)) -ErrorAction SilentlyContinue)
    if (($forceSmart -notmatch '^(1|true|yes)$') -and ($existingSmart.Count -gt 0)) {
      Write-Log ("Skipping smart-sim-date (found {0} existing {1} artifacts; set DAILY_FORCE_SMARTSIM_DATE=1 to rerun)" -f $existingSmart.Count, $SmartSimOutPrefix)
    } else {

    # Generate team advanced stats priors (pace/ratings) as-of today to avoid any future leakage.
    try {
      $dts = [datetime]::Parse($Date)
      $seasonY = if ($dts.Month -ge 7) { $dts.Year + 1 } else { $dts.Year }
      Write-Log ("Updating team advanced stats priors (season={0}, as_of={1})" -f $seasonY, $Date)
      $rcAdv = Invoke-PyMod -plist @('-m','nba_betting.cli','fetch-advanced-stats','--season', $seasonY, '--as-of', $Date)
      Write-Log ("fetch-advanced-stats exit code: {0}" -f $rcAdv)
    } catch {
      Write-Log ("fetch-advanced-stats failed (non-fatal): {0}" -f $_.Exception.Message)
    }

    $nSmart = $env:DAILY_SMARTSIM_NSIMS
    if ($null -eq $nSmart -or $nSmart -eq '') { $nSmart = '2000' }
    $maxSmart = $env:DAILY_SMARTSIM_MAX_GAMES
    $doOverwrite = $env:DAILY_SMARTSIM_OVERWRITE
    if ($null -eq $doOverwrite -or $doOverwrite -eq '') { $doOverwrite = '0' }
    $plist = @('-m','nba_betting.cli','smart-sim-date','--date', $Date, '--n-sims', $nSmart, '--roster-mode', $SmartSimRosterMode, '--out-prefix', $SmartSimOutPrefix)

    # Optional: parallelize per-game SmartSim jobs (matches predict-props SmartSim workers behavior)
    try {
      if ($null -ne $SmartSimWorkers -and $SmartSimWorkers -match '^\d+$' -and [int]$SmartSimWorkers -gt 1) {
        Write-Log ("Using SmartSim parallel workers for smart-sim-date: {0}" -f $SmartSimWorkers)
        $plist += @('--workers', $SmartSimWorkers)
      }
    } catch {}

    if ($doOverwrite -match '^(1|true|yes)$') { $plist += @('--overwrite') }
    if ($null -ne $maxSmart -and $maxSmart -ne '') { $plist += @('--max-games', $maxSmart) }
    Write-Log ("Running SmartSim slate for {0} (n_sims={1}, roster_mode={2}, out_prefix={3})" -f $Date, $nSmart, $SmartSimRosterMode, $SmartSimOutPrefix)
    $rcSmart = Invoke-PyMod -plist $plist
    Write-Log ("smart-sim-date exit code: {0}" -f $rcSmart)
    }
  } else {
    Write-Log 'Skipping smart-sim-date (DAILY_SKIP_SMARTSIM=1)'
  }
} catch {
  Write-Log ("smart-sim-date failed (non-fatal): {0}" -f $_.Exception.Message)
}

# 7.25) Build cards sim detail snapshot artifact for lazy-loaded box score detail (non-fatal)
try {
  $skipCardsSimDetail = $env:DAILY_SKIP_CARDS_SIM_DETAIL
  if ($null -ne $skipCardsSimDetail -and $skipCardsSimDetail -match '^(1|true|yes)$') {
    Write-Log 'Skipping cards sim detail snapshot build (DAILY_SKIP_CARDS_SIM_DETAIL=1)'
  } else {
    $cardsSimPropsSource = [string]$env:DAILY_CARDS_SIM_DETAIL_PROPS_SOURCE
    if ([string]::IsNullOrWhiteSpace($cardsSimPropsSource)) { $cardsSimPropsSource = 'auto' }
    $cardsSimPropsSource = $cardsSimPropsSource.Trim().ToLowerInvariant()
    if (@('auto', 'source', 'runtime', 'snapshot') -notcontains $cardsSimPropsSource) { $cardsSimPropsSource = 'auto' }
    $cardsSimDetailOut = Join-Path $RepoRoot ("data/processed/cards_sim_detail_{0}.json" -f $Date)
    try { if (Test-Path $cardsSimDetailOut) { Remove-Item $cardsSimDetailOut -Force -ErrorAction SilentlyContinue } } catch { }
    Write-Log ("Building cards sim detail snapshot for {0} -> {1} (props_source={2})" -f $Date, $cardsSimDetailOut, $cardsSimPropsSource)

    $tmpPyCardsSim = Join-Path $LogPath ("build_cards_sim_detail_{0}.py" -f $Stamp)
    $pyCardsSim = @"
import json
import sys
from pathlib import Path

repo_root = Path(r"{REPO_PLACEHOLDER}")
if str(repo_root) not in sys.path:
  sys.path.insert(0, str(repo_root))

import app

date_str = r"{DATE_PLACEHOLDER}"
out_path = Path(r"{OUT_PLACEHOLDER}")
props_source = r"{PROPS_SOURCE_PLACEHOLDER}"

client = app.app.test_client()
resp = client.get(f"/api/cards?date={date_str}&include_players=1&props_source={props_source}")
try:
  payload = resp.get_json() if resp is not None else None
except Exception:
  payload = None

games_out = []
if isinstance(payload, dict):
  for game in (payload.get("games") or []):
    if not isinstance(game, dict):
      continue
    sim = game.get("sim") if isinstance(game.get("sim"), dict) else {}
    players = sim.get("players") if isinstance(sim.get("players"), dict) else {}
    missing = sim.get("missing_prop_players") if isinstance(sim.get("missing_prop_players"), dict) else {}
    injuries = sim.get("injuries") if isinstance(sim.get("injuries"), dict) else {}
    summary = sim.get("players_summary") if isinstance(sim.get("players_summary"), dict) else {
      "home": len(players.get("home") or []),
      "away": len(players.get("away") or []),
      "missing_home": len(missing.get("home") or []),
      "missing_away": len(missing.get("away") or []),
      "injured_home": len(injuries.get("home") or []),
      "injured_away": len(injuries.get("away") or []),
    }
    games_out.append({
      "home_tri": game.get("home_tri"),
      "away_tri": game.get("away_tri"),
      "sim": {
        "players_summary": dict(summary),
        "players": {
          "home": [row for row in (players.get("home") or []) if isinstance(row, dict)],
          "away": [row for row in (players.get("away") or []) if isinstance(row, dict)],
        },
        "missing_prop_players": {
          "home": [row for row in (missing.get("home") or []) if isinstance(row, dict)],
          "away": [row for row in (missing.get("away") or []) if isinstance(row, dict)],
        },
        "injuries": {
          "home": [row for row in (injuries.get("home") or []) if isinstance(row, dict)],
          "away": [row for row in (injuries.get("away") or []) if isinstance(row, dict)],
        },
      },
    })

out = {
  "date": date_str,
  "games": games_out,
}
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
print("OK")
"@
    $pyCardsSim = $pyCardsSim.Replace('{DATE_PLACEHOLDER}', $Date)
    $pyCardsSim = $pyCardsSim.Replace('{REPO_PLACEHOLDER}', $RepoRoot)
    $pyCardsSim = $pyCardsSim.Replace('{OUT_PLACEHOLDER}', $cardsSimDetailOut)
    $pyCardsSim = $pyCardsSim.Replace('{PROPS_SOURCE_PLACEHOLDER}', $cardsSimPropsSource)
    Set-Content -Path $tmpPyCardsSim -Value $pyCardsSim -Encoding UTF8
    $outCardsSim = & $Python $tmpPyCardsSim 2>&1 | Tee-Object -FilePath $LogFile -Append
    if ($outCardsSim -match 'OK') { Write-Log ("Wrote {0}" -f $cardsSimDetailOut) } else { Write-Log ("Cards sim detail snapshot build returned: {0}" -f $outCardsSim) }
  }
} catch {
  Write-Log ("Cards sim detail snapshot build failed (non-fatal): {0}" -f $_.Exception.Message)
}

# 7.255) Build season betting-card manifest/day artifacts for Render consumption
try {
  $skipBettingCardPrebuild = $env:DAILY_SKIP_SEASON_BETTING_CARD_PREBUILD
  if ($null -ne $skipBettingCardPrebuild -and $skipBettingCardPrebuild -match '^(1|true|yes)$') {
    Write-Log 'Skipping season betting-card prebuild (DAILY_SKIP_SEASON_BETTING_CARD_PREBUILD=1)'
  } else {
    $bettingCardProfile = [string]$env:DAILY_SEASON_BETTING_CARD_PROFILE
    if ([string]::IsNullOrWhiteSpace($bettingCardProfile)) { $bettingCardProfile = 'retuned' }
    $bettingCardProfile = $bettingCardProfile.Trim().ToLowerInvariant()
    $bettingCardSeason = Resolve-NbaSeasonYear -DateValue $Date
    $manifestOut = Join-Path $RepoRoot ("data/processed/season_betting_card_manifest_{0}_{1}_{2}.json" -f $bettingCardSeason, $bettingCardProfile, $Date)
    $manifestGenericOut = Join-Path $RepoRoot ("data/processed/season_betting_card_manifest_{0}_{1}.json" -f $bettingCardSeason, $bettingCardProfile)
    $dayOut = Join-Path $RepoRoot ("data/processed/season_betting_card_day_{0}_{1}_{2}.json" -f $bettingCardSeason, $bettingCardProfile, $Date)
    $dayInsightsOut = Join-Path $RepoRoot ("data/processed/season_betting_card_day_{0}_{1}_{2}_insights.json" -f $bettingCardSeason, $bettingCardProfile, $Date)
    Write-Log ("Building season betting-card artifacts for season={0}, date={1}, profile={2}" -f $bettingCardSeason, $Date, $bettingCardProfile)

    $tmpPyBettingCard = Join-Path $LogPath ("build_season_betting_card_artifacts_{0}.py" -f $Stamp)
    $pyBettingCard = @(
      'import json',
      'import shutil',
      'import sys',
      'from pathlib import Path',
      '',
      'repo_root = Path(r"{REPO_PLACEHOLDER}")',
      'if str(repo_root) not in sys.path:',
      '    sys.path.insert(0, str(repo_root))',
      '',
      'import app',
      '',
      'season = int(r"{SEASON_PLACEHOLDER}")',
      'profile = r"{PROFILE_PLACEHOLDER}"',
      'date_str = r"{DATE_PLACEHOLDER}"',
      'manifest_out = Path(r"{MANIFEST_OUT_PLACEHOLDER}")',
      'manifest_generic_out = Path(r"{MANIFEST_GENERIC_OUT_PLACEHOLDER}")',
      'day_out = Path(r"{DAY_OUT_PLACEHOLDER}")',
      'day_insights_out = Path(r"{DAY_INSIGHTS_OUT_PLACEHOLDER}")',
      '',
      'client = app.app.test_client()',
      '',
      'def fetch_json(path: str) -> dict:',
      '    resp = client.get(path)',
      '    if resp is None:',
      '        raise RuntimeError(f"No response for {path}")',
      '    if int(resp.status_code or 0) >= 400:',
      '        raise RuntimeError(f"{path} returned status {resp.status_code}")',
      '    payload = resp.get_json(silent=True)',
      '    if not isinstance(payload, dict):',
      '        raise RuntimeError(f"{path} returned non-dict payload")',
      '    return payload',
      '',
      'manifest_payload = fetch_json(f"/api/season/{season}/betting-card?profile={profile}&date={date_str}")',
      'day_payload = fetch_json(f"/api/season/{season}/betting-card/day/{date_str}?profile={profile}")',
      'day_insights_payload = fetch_json(f"/api/season/{season}/betting-card/day/{date_str}?profile={profile}&include_prop_insights=1")',
      '',
      'for out_path, payload in (',
      '    (manifest_out, manifest_payload),',
      '    (day_out, day_payload),',
      '    (day_insights_out, day_insights_payload),',
      '):',
      '    out_path.parent.mkdir(parents=True, exist_ok=True)',
      '    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")',
      '',
      'shutil.copyfile(manifest_out, manifest_generic_out)',
      'print("OK")'
    ) -join "`r`n"
    $pyBettingCard = $pyBettingCard.Replace('{REPO_PLACEHOLDER}', $RepoRoot)
    $pyBettingCard = $pyBettingCard.Replace('{SEASON_PLACEHOLDER}', [string]$bettingCardSeason)
    $pyBettingCard = $pyBettingCard.Replace('{PROFILE_PLACEHOLDER}', $bettingCardProfile)
    $pyBettingCard = $pyBettingCard.Replace('{DATE_PLACEHOLDER}', $Date)
    $pyBettingCard = $pyBettingCard.Replace('{MANIFEST_OUT_PLACEHOLDER}', $manifestOut)
    $pyBettingCard = $pyBettingCard.Replace('{MANIFEST_GENERIC_OUT_PLACEHOLDER}', $manifestGenericOut)
    $pyBettingCard = $pyBettingCard.Replace('{DAY_OUT_PLACEHOLDER}', $dayOut)
    $pyBettingCard = $pyBettingCard.Replace('{DAY_INSIGHTS_OUT_PLACEHOLDER}', $dayInsightsOut)
    Set-Content -Path $tmpPyBettingCard -Value $pyBettingCard -Encoding UTF8
    $outBettingCard = & $Python $tmpPyBettingCard 2>&1 | Tee-Object -FilePath $LogFile -Append
    if ($outBettingCard -notmatch 'OK') {
      throw ("Season betting-card artifact build returned: {0}" -f $outBettingCard)
    }

    foreach ($artifactPath in @($manifestOut, $manifestGenericOut, $dayOut, $dayInsightsOut)) {
      if (-not (Test-Path $artifactPath)) {
        throw ("Missing season betting-card artifact after build: {0}" -f $artifactPath)
      }
      Write-Log ("Wrote {0}" -f $artifactPath)
    }
  }
} catch {
  Write-Log ("Season betting-card prebuild failed: {0}" -f $_.Exception.Message)
  throw
}

# 7.26) Validate cards + betting-card portfolio render payloads (fail fast)
try {
  $skipCardsRenderValidation = $env:DAILY_SKIP_CARDS_PORTFOLIO_RENDER_VALIDATION
  if ($null -ne $skipCardsRenderValidation -and $skipCardsRenderValidation -match '^(1|true|yes)$') {
    Write-Log 'Skipping cards portfolio render validation (DAILY_SKIP_CARDS_PORTFOLIO_RENDER_VALIDATION=1)'
  } else {
    $cardsRenderTimeout = $env:DAILY_CARDS_PORTFOLIO_RENDER_TIMEOUT_SEC
    if ($null -eq $cardsRenderTimeout -or $cardsRenderTimeout -eq '') { $cardsRenderTimeout = '180' }
    try { $cardsRenderTimeoutInt = [int]$cardsRenderTimeout } catch { $cardsRenderTimeoutInt = 180 }
    if ($cardsRenderTimeoutInt -lt 30) { $cardsRenderTimeoutInt = 30 }
    Write-Log ("Validating cards + betting-card portfolio render inputs for {0}" -f $Date)
    $rcCardsRender = Invoke-PyModWithTimeout -plist @(
      'tools/validate_cards_portfolio_render.py',
      '--date', $Date,
      '--profile', 'retuned',
      '--props-source', 'source'
    ) -TimeoutSeconds $cardsRenderTimeoutInt -Label 'validate_cards_portfolio_render'
    Write-Log ("validate_cards_portfolio_render exit code: {0}" -f $rcCardsRender)
    if ($rcCardsRender -ne 0) {
      throw ("cards portfolio render validation failed (exit={0})" -f $rcCardsRender)
    }
  }
} catch {
  Write-Log ("Cards portfolio render validation failed: {0}" -f $_.Exception.Message)
  throw
}

# 8) End-to-end artifact validation (writes data/processed/daily_artifacts_<date>.json)
try {
  $fail = $env:DAILY_FAIL_ON_MISSING_ARTIFACTS
  if ($null -eq $fail -or $fail -eq '') { $fail = '1' }
  $reqOdds = $env:DAILY_REQUIRE_ODDS
  if ($null -eq $reqOdds -or $reqOdds -eq '') { $reqOdds = '0' }
  $reqSmart = $env:DAILY_REQUIRE_SMARTSIM
  if ($null -eq $reqSmart -or $reqSmart -eq '') {
    # If SmartSim was intentionally skipped, don't require SmartSim artifacts by default.
    $skipSmartNow = $env:DAILY_SKIP_SMARTSIM
    if ($null -ne $skipSmartNow -and $skipSmartNow -match '^(1|true|yes)$') { $reqSmart = '0' } else { $reqSmart = '1' }
  }
  $reqPropsLines = $env:DAILY_REQUIRE_PROPS_LINES
  if ($null -eq $reqPropsLines -or $reqPropsLines -eq '') { $reqPropsLines = '1' }
  $reqRot = $env:DAILY_REQUIRE_ROTATIONS_ESPN
  if ($null -eq $reqRot -or $reqRot -eq '') { $reqRot = '0' }

    $env:REPO_ROOT = $RepoRoot
    $env:FAIL_ON_MISSING = $fail
    $env:REQUIRE_ODDS = $reqOdds
    $env:REQUIRE_SMARTSIM = $reqSmart
    $env:REQUIRE_PROPS_LINES = $reqPropsLines
    $env:REQUIRE_ROTATIONS = $reqRot
    if ($null -eq $env:ROTATIONS_MIN_COVERAGE -or $env:ROTATIONS_MIN_COVERAGE -eq '') { $env:ROTATIONS_MIN_COVERAGE = '0.70' }
    Write-Log ("Validating daily artifacts (require_odds={0}, require_smartsim={1}, require_props_lines={2}, require_rotations_espn={3})" -f $reqOdds, $reqSmart, $reqPropsLines, $reqRot)
    $outV = Invoke-PyMod -plist @(
    'tools/validate_daily_artifacts.py',
    '--repo-root', $RepoRoot,
    '--date', $Date,
    '--yesterday', $yesterday,
    '--rotations-min-coverage', $env:ROTATIONS_MIN_COVERAGE
    )
  if ($outV -ne 0) {
    Write-Log ("Daily artifact validation failed (exit={0})" -f $outV)
    if ($fail -match '^(1|true|yes)$') { throw "daily artifacts missing" }
  } else {
    Write-Log 'Daily artifact validation OK'
  }
} catch {
  Write-Log ("Daily artifact validation block failed: {0}" -f $_.Exception.Message)
  throw
}

# 8.1) Player availability audits (fail loudly)
# - Ensures SmartSim JSON includes all expected props_predictions players
# - Ensures no stale injury exclusions conflict with playing_today
try {
  $skipAud = $env:DAILY_SKIP_PLAYER_AUDITS
  if ($null -eq $skipAud -or $skipAud -notmatch '^(1|true|yes)$') {
    Write-Log ("Running player audits for {0}" -f $Date)

    $skipSmartAud = $env:DAILY_SKIP_SMARTSIM
    if ($null -ne $skipSmartAud -and $skipSmartAud -match '^(1|true|yes)$') {
      Write-Log 'Skipping SmartSim audits (DAILY_SKIP_SMARTSIM=1)'
    } else {
      $rcCov = Invoke-PyMod -plist @('tools/audit_smart_sim_player_coverage.py','--date', $Date)
      Write-Log ("audit_smart_sim_player_coverage exit code: {0}" -f $rcCov)
      if ($rcCov -ne 0) { throw "SmartSim player coverage audit failed (exit=$rcCov)" }

      $rcMin = Invoke-PyMod -plist @('tools/audit_smart_sim_minutes.py','--date', $Date)
      Write-Log ("audit_smart_sim_minutes exit code: {0}" -f $rcMin)
      if ($rcMin -ne 0) { throw "SmartSim minutes audit failed (exit=$rcMin)" }
    }

    $rcStale = Invoke-PyMod -plist @('tools/audit_stale_exclusions_today.py','--date', $Date)
    Write-Log ("audit_stale_exclusions_today exit code: {0}" -f $rcStale)
    if ($rcStale -ne 0) { throw "Stale exclusions audit failed (exit=$rcStale)" }

    $rcInjC = Invoke-PyMod -plist @('tools/audit_injuries_counts_consistency.py','--date', $Date)
    Write-Log ("audit_injuries_counts_consistency exit code: {0}" -f $rcInjC)
    if ($rcInjC -ne 0) { throw "injuries_counts consistency audit failed (exit=$rcInjC)" }
  } else {
    Write-Log 'Skipping player audits (DAILY_SKIP_PLAYER_AUDITS=1)'
  }
} catch {
  Write-Log ("Player audits failed: {0}" -f $_.Exception.Message)
  throw
}

# Optional: Live Lens tuning (optimize adjustments from logged signals + recon actuals)
# Writes data/processed/live_lens_tuning_override.json when enough signal-backed bets exist.
try {
  $LiveLensDir = $env:NBA_LIVE_LENS_DIR
  if ($null -eq $LiveLensDir -or $LiveLensDir -eq '') { $LiveLensDir = $env:LIVE_LENS_DIR }
  if ($null -eq $LiveLensDir -or $LiveLensDir -eq '') { $LiveLensDir = (Join-Path $RepoRoot 'data/processed') }
  if (-not (Test-Path $LiveLensDir)) { New-Item -ItemType Directory -Path $LiveLensDir | Out-Null }

  # Ensure downstream Python tools read/write Live Lens artifacts in the same place.
  $env:NBA_LIVE_LENS_DIR = $LiveLensDir

  # Optional: fetch recent Live Lens logs from a remote server (e.g., Render) before tuning.
  # This is ON by default (safe/no-op when remote is unreachable or artifacts are missing).
  # Disable via: DAILY_FETCH_REMOTE_LIVE_LENS=0
  $fetchRemote = $env:DAILY_FETCH_REMOTE_LIVE_LENS
  if ($null -eq $fetchRemote -or $fetchRemote -eq '') { $fetchRemote = '1' }
  if ($null -ne $fetchRemote -and $fetchRemote -match '^(1|true|yes)$') {
    try {
      # Prefer shared env var used by cron tooling; fall back to script param.
      $remote = $env:NBA_BETTING_BASE_URL
      if ($null -eq $remote -or $remote -eq '') { $remote = $RemoteBaseUrl }
      if ($null -ne $remote -and $remote -ne '') {
        $remote = $remote.TrimEnd('/')

        # Fast preflight so we don't hang on per-date timeouts when Render is down.
        $remoteOk = $true
        try {
          $healthUrl = "{0}/health" -f $remote
          $h = Invoke-WebRequest -Uri $healthUrl -TimeoutSec 10 -UseBasicParsing -ErrorAction Stop
          if ($null -eq $h -or $h.StatusCode -lt 200 -or $h.StatusCode -ge 300) { $remoteOk = $false }
        } catch {
          $remoteOk = $false
        }

        if (-not $remoteOk) {
          Write-Log ("Live Lens: remote health check failed ({0}); skipping fetch" -f $remote)
        } else {
          $forceFetch = $env:DAILY_FORCE_FETCH_REMOTE_LIVE_LENS
          $doForce = ($null -ne $forceFetch -and $forceFetch -match '^(1|true|yes)$')

          $lookbackDays = 14
          try {
            $lb = $env:DAILY_LIVE_LENS_LOOKBACK_DAYS
            if ($null -ne $lb -and $lb -ne '') { $lookbackDays = [int]$lb }
          } catch { $lookbackDays = 14 }
          if ($lookbackDays -lt 1) { $lookbackDays = 1 }
          if ($lookbackDays -gt 60) { $lookbackDays = 60 }

          $endD = [datetime]::ParseExact($yesterday, 'yyyy-MM-dd', $null)
          $startD = $endD.AddDays(-($lookbackDays - 1))

          function Get-RemoteContentLength {
            param([string]$url)
            try {
              $hh = Invoke-WebRequest -Uri $url -Method Head -TimeoutSec 15 -UseBasicParsing -ErrorAction Stop
              $cl = $hh.Headers['Content-Length']
              if ($null -ne $cl -and $cl -ne '') { return [int64]$cl }
            } catch { }
            return $null
          }

          Write-Log ("Live Lens: reconciling remote JSONLs {0}..{1} -> {2} (remote={3})" -f $startD.ToString('yyyy-MM-dd'), $endD.ToString('yyyy-MM-dd'), $LiveLensDir, $remote)

          for ($d = $startD; $d -le $endD; $d = $d.AddDays(1)) {
            $ds = $d.ToString('yyyy-MM-dd')
            $sigOut = Join-Path $LiveLensDir ("live_lens_signals_{0}.jsonl" -f $ds)
            $projOut = Join-Path $LiveLensDir ("live_lens_projections_{0}.jsonl" -f $ds)

            try {
              $u1 = "{0}/api/download_live_lens_signals?date={1}" -f $remote, $ds
              $needSig = ($doForce -or -not (Test-Path $sigOut))
              if (-not $needSig) {
                try {
                  $localBytes = (Get-Item $sigOut).Length
                  $remoteBytes = Get-RemoteContentLength -url $u1
                  if ($null -ne $remoteBytes -and $remoteBytes -gt [int64]$localBytes) { $needSig = $true }
                } catch { }
              }

              if ($needSig) {
                $tmp = ("{0}.tmp_download" -f $sigOut)
                try {
                  Invoke-WebRequest -Uri $u1 -OutFile $tmp -TimeoutSec 30 -UseBasicParsing -ErrorAction Stop | Out-Null
                  if (Test-Path $tmp) {
                    Move-Item -Path $tmp -Destination $sigOut -Force
                    Write-Log ("Live Lens: synced signals {0}" -f $ds)
                  }
                } finally {
                  try { if (Test-Path $tmp) { Remove-Item $tmp -Force -ErrorAction SilentlyContinue } } catch { }
                }
              }
            } catch {
              Write-Log ("Live Lens: signals missing/failed for {0} (non-fatal)" -f $ds)
            }

            try {
              $u2 = "{0}/api/download_live_lens_projections?date={1}" -f $remote, $ds
              $needProj = ($doForce -or -not (Test-Path $projOut))
              if (-not $needProj) {
                try {
                  $localBytesP = (Get-Item $projOut).Length
                  $remoteBytesP = Get-RemoteContentLength -url $u2
                  if ($null -ne $remoteBytesP -and $remoteBytesP -gt [int64]$localBytesP) { $needProj = $true }
                } catch { }
              }

              if ($needProj) {
                $tmp2 = ("{0}.tmp_download" -f $projOut)
                try {
                  Invoke-WebRequest -Uri $u2 -OutFile $tmp2 -TimeoutSec 30 -UseBasicParsing -ErrorAction Stop | Out-Null
                  if (Test-Path $tmp2) {
                    Move-Item -Path $tmp2 -Destination $projOut -Force
                    Write-Log ("Live Lens: synced projections {0}" -f $ds)
                  }
                } finally {
                  try { if (Test-Path $tmp2) { Remove-Item $tmp2 -Force -ErrorAction SilentlyContinue } } catch { }
                }
              }
            } catch {
              # Projections are optional; do not log loudly.
            }
          }
        }
      } else {
        Write-Log 'Live Lens: DAILY_FETCH_REMOTE_LIVE_LENS=1 but RemoteBaseUrl is empty; skipping fetch'
      }
    } catch {
      Write-Log ("Live Lens: remote fetch failed (non-fatal): {0}" -f $_.Exception.Message)
    }
  }

  $sigPath = Join-Path $LiveLensDir ("live_lens_signals_{0}.jsonl" -f $yesterday)
  if (Test-Path $sigPath) {
    Write-Log ("Live Lens tune: signals present for {0}; running optimizer" -f $yesterday)
    $lookbackDays = 14
    try {
      $lb2 = $env:DAILY_LIVE_LENS_LOOKBACK_DAYS
      if ($null -ne $lb2 -and $lb2 -ne '') { $lookbackDays = [int]$lb2 }
    } catch { $lookbackDays = 14 }
    if ($lookbackDays -lt 1) { $lookbackDays = 1 }
    if ($lookbackDays -gt 60) { $lookbackDays = 60 }
    $rcLens = Invoke-PyMod -plist @(
      'tools/daily_live_lens_tune.py',
      '--end', $yesterday,
      '--lookback-days', $lookbackDays,
      '--min-bets', '10',
      '--write-override'
    )
    Write-Log ("daily_live_lens_tune exit code: {0}" -f $rcLens)

    # Optional: player prop threshold tune (from settled prop outcomes)
    try {
      $skipPropTune = $env:DAILY_SKIP_LIVE_LENS_PROP_TUNE
      if ($null -eq $skipPropTune -or $skipPropTune -notmatch '^(1|true|yes)$') {
        $rp = Join-Path $RepoRoot ("data/processed/recon_props_{0}.csv" -f $yesterday)
        if (Test-Path $rp) {
          Write-Log ("Live Lens prop tune: recon_props present for {0}; optimizing player_prop thresholds" -f $yesterday)
          $rcPropTune = Invoke-PyMod -plist @(
            'tools/optimize_live_lens_player_prop_thresholds.py',
            '--start', $yesterday,
            '--end', $yesterday,
            '--min-bets', '30',
            '--also-sigma',
            '--sigma-per-stat',
            '--sigma-min-bets-per-stat', '15',
            '--write-override'
          )
          Write-Log ("optimize_live_lens_player_prop_thresholds exit code: {0}" -f $rcPropTune)
        } else {
          Write-Log ("Live Lens prop tune: no recon_props for {0}; skipping" -f $yesterday)
        }
      } else {
        Write-Log 'Skipping Live Lens prop tune (DAILY_SKIP_LIVE_LENS_PROP_TUNE=1)'
      }
    } catch {
      Write-Log ("Live Lens prop tune failed (non-fatal): {0}" -f $_.Exception.Message)
    }

    # Optional: ROI report (settle logged signals into units)
    try {
      $skipRoi = $env:DAILY_SKIP_LIVE_LENS_ROI
      if ($null -eq $skipRoi -or $skipRoi -notmatch '^(1|true|yes)$') {
        Write-Log ("Live Lens ROI: generating report for {0}" -f $yesterday)
        $rcRoi = Invoke-PyMod -plist @(
          'tools/daily_live_lens_roi.py',
          '--date', $yesterday
        )
        Write-Log ("daily_live_lens_roi exit code: {0}" -f $rcRoi)
      } else {
        Write-Log 'Skipping Live Lens ROI (DAILY_SKIP_LIVE_LENS_ROI=1)'
      }
    } catch {
      Write-Log ("Live Lens ROI failed (non-fatal): {0}" -f $_.Exception.Message)
    }
  } else {
    Write-Log ("Live Lens tune: no signals file for {0}; skipping" -f $yesterday)
  }
} catch {
  Write-Log ("Live Lens tune failed (non-fatal): {0}" -f $_.Exception.Message)
}

# 8.9) Future-safe look-ahead artifacts for Date+N (default N=1)
# This gives Render/UI a forward slate to use when the requested date has no current-day cards yet.
$LookAheadDates = @()
try {
  if ($LookAheadDays -gt 0) {
    Write-Log ("Look-ahead enabled: building {0} future day(s)" -f $LookAheadDays)
    $baseDate = [datetime]::ParseExact($Date, 'yyyy-MM-dd', $null)
    for ($i = 1; $i -le $LookAheadDays; $i++) {
      $futureDate = $baseDate.AddDays($i).ToString('yyyy-MM-dd')
      $okFuture = Invoke-LookAheadDailyUpdateJob -TargetDate $futureDate
      if ($okFuture) {
        $LookAheadDates += $futureDate
      } else {
        Write-Log ("Look-ahead skipped push/commit for {0} due to incomplete artifacts" -f $futureDate)
      }
    }
  } else {
    Write-Log 'Look-ahead disabled (LookAheadDays=0)'
  }
} catch {
  Write-Log ("Look-ahead block failed (non-fatal): {0}" -f $_.Exception.Message)
}

# Simple retention: keep last 21 local_daily_update_* logs
Get-ChildItem -Path $LogPath -Filter 'local_daily_update_*.log' | Sort-Object LastWriteTime -Descending | Select-Object -Skip 21 | ForEach-Object { Remove-Item $_.FullName -ErrorAction SilentlyContinue }

# Ensure push to Git is the final action of the script when enabled
$FinalGitPushAttempted = $false
$FinalGitPushSucceeded = $false
if (-not $GitPush) {
  Write-Log 'Local daily update complete (no Git push requested).'
} else {
  # Use standardized commit script to stage and push only date-scoped processed artifacts
  try {
    Write-Log 'Git: committing processed artifacts via scripts/commit_processed.ps1 (yesterday, today, then look-ahead dates)'
    $commitScript = Join-Path $RepoRoot 'scripts/commit_processed.ps1'
    if (Test-Path $commitScript) {
      Remove-StaleGitLock
      # First, commit yesterday's finals/reconcile outputs without push
      & powershell -NoProfile -ExecutionPolicy Bypass -File $commitScript -Date $yesterday -IncludeJson | Tee-Object -FilePath $LogFile -Append | Out-Null
      # Then, commit today's predictions/edges/odds (no push yet)
      Remove-StaleGitLock
      & powershell -NoProfile -ExecutionPolicy Bypass -File $commitScript -Date $Date -IncludeJson | Tee-Object -FilePath $LogFile -Append | Out-Null

      # Finally, commit any validated look-ahead dates (tomorrow, etc.) so a single git push
      # can publish both current-day and future-day artifacts together.
      foreach ($futureDate in ($LookAheadDates | Select-Object -Unique)) {
        try {
          Remove-StaleGitLock
          Write-Log ("Git: committing look-ahead artifacts for {0}" -f $futureDate)
          & powershell -NoProfile -ExecutionPolicy Bypass -File $commitScript -Date $futureDate -IncludeJson | Tee-Object -FilePath $LogFile -Append | Out-Null
        } catch {
          Write-Log ("Git: look-ahead commit failed for {0}: {1}" -f $futureDate, $_.Exception.Message)
        }
      }

      # Additionally stage yearly PBP metrics CSV if present/changed.
      try {
        $metricsYear = ($yesterday.Substring(0,4))
        $metricsPath = Join-Path $RepoRoot ("data/processed/pbp_metrics_daily_{0}.csv" -f $metricsYear)
        if (Test-Path $metricsPath) {
          & git add -- $metricsPath 2>&1 | Tee-Object -FilePath $LogFile -Append | Out-Null
          $mchanged = & git diff --cached --name-only -- $metricsPath
          if ($mchanged) {
            $msg2 = "data(processed): update pbp metrics daily ($yesterday)"
            Remove-StaleGitLock
            & git commit -m $msg2 2>&1 | Tee-Object -FilePath $LogFile -Append | Out-Null
            Write-Log 'Git: committed pbp_metrics_daily update'
          } else {
            Write-Log 'Git: no changes in pbp_metrics_daily to commit'
          }
        }
      } catch { Write-Log ("Git: commit pbp_metrics_daily failed: {0}" -f $_.Exception.Message) }

      try {
        Write-Log 'Git: pushing accumulated daily update commits'
        Remove-StaleGitLock
        $FinalGitPushAttempted = $true
        $rcFinalPush = Invoke-LoggedNativeCommand -FilePath 'git' -ArgumentList @('push')
        if ($rcFinalPush -eq 0) {
          $FinalGitPushSucceeded = $true
          Write-Log 'Git: final push succeeded'
        } else {
          Write-Log ("Git: final push failed (exit={0}); attempting pull --rebase origin main then retry" -f $rcFinalPush)
          Remove-StaleGitLock
          $rcFinalPull = Invoke-LoggedNativeCommand -FilePath 'git' -ArgumentList @('pull', '--rebase', 'origin', 'main')
          if ($rcFinalPull -ne 0) {
            Write-Log ("Git: final pull --rebase failed (exit={0})" -f $rcFinalPull)
          } else {
            Remove-StaleGitLock
            $rcFinalPushRetry = Invoke-LoggedNativeCommand -FilePath 'git' -ArgumentList @('push', 'origin', 'HEAD:main')
            if ($rcFinalPushRetry -eq 0) {
              $FinalGitPushSucceeded = $true
              Write-Log 'Git: final push succeeded after rebase retry'
            } else {
              Write-Log ("Git: final push retry failed (exit={0})" -f $rcFinalPushRetry)
            }
          }
        }
      } catch {
        Write-Log ("Git: final push failed: {0}" -f $_.Exception.Message)
      }
    } else {
      Write-Log 'Commit script missing; falling back to broad staging (data/processed)'
      Remove-StaleGitLock
      & git add -- data data\processed 2>&1 | Tee-Object -FilePath $LogFile -Append | Out-Null
    # Legacy root predictions.csv intentionally not staged (Render UI reads date-scoped processed files)
      $cached = & git diff --cached --name-only
      if ($cached) {
        $msg = "local daily: $Date (predictions/odds/props)"
        Remove-StaleGitLock
        & git commit -m $msg 2>&1 | Tee-Object -FilePath $LogFile -Append | Out-Null
        $rcFallbackPull = Invoke-LoggedNativeCommand -FilePath 'git' -ArgumentList @('pull', '--rebase')
        if ($rcFallbackPull -ne 0) {
          throw "git pull --rebase failed (exit=$rcFallbackPull)"
        }
        $FinalGitPushAttempted = $true
        $rcFallbackPush = Invoke-LoggedNativeCommand -FilePath 'git' -ArgumentList @('push')
        if ($rcFallbackPush -eq 0) {
          $FinalGitPushSucceeded = $true
          Write-Log 'Git: final push succeeded'
        } else {
          throw "git push failed (exit=$rcFallbackPush)"
        }
      } else {
        Write-Log 'Git: no staged changes; skipping push'
      }
    }
  } catch {
    Write-Log ("Git push failed: {0}" -f $_.Exception.Message)
  }
}

# Optional: emit today's pregame prop recommendations into the Live Lens signal stream.
# This runs AFTER the git push so the remote /api/cards reflects the newly pushed artifacts.
# Disable via: DAILY_LOG_PREGAME_PROP_SIGNALS=0
try {
  $emitPregame = $env:DAILY_LOG_PREGAME_PROP_SIGNALS
  if ($null -eq $emitPregame -or $emitPregame -eq '') { $emitPregame = '1' }

  $allowNonToday = $env:DAILY_LOG_PREGAME_PROP_SIGNALS_ALLOW_NON_TODAY
  if ($null -eq $allowNonToday -or $allowNonToday -eq '') { $allowNonToday = '0' }
  $todayLocal = Resolve-NbaSlateDate

  if ($NoSlateDay) {
    Write-Log ("Pregame signals: no slate for {0}; skipping" -f $Date)
  } elseif ($allowNonToday -notmatch '^(1|true|yes)$' -and $Date -ne $todayLocal) {
    Write-Log ("Pregame signals: Date={0} is not today ({1}); skipping (set DAILY_LOG_PREGAME_PROP_SIGNALS_ALLOW_NON_TODAY=1 to override)" -f $Date, $todayLocal)
  } elseif ($null -ne $emitPregame -and $emitPregame -match '^(1|true|yes)$') {
    $remote2 = $env:NBA_BETTING_BASE_URL
    if ($null -eq $remote2 -or $remote2 -eq '') { $remote2 = $RemoteBaseUrl }

    if ($null -ne $remote2 -and $remote2 -ne '') {
      $remote2 = $remote2.TrimEnd('/')

      if ($GitPush -and $FinalGitPushAttempted -and -not $FinalGitPushSucceeded) {
        Write-Log 'Pregame signals: skipping because final git push did not succeed'
        return
      }

      $remoteOk2 = $true
      try {
        $healthUrl2 = "{0}/health" -f $remote2
        $h2 = Invoke-WebRequest -Uri $healthUrl2 -TimeoutSec 10 -UseBasicParsing -ErrorAction Stop
        if ($null -eq $h2 -or $h2.StatusCode -lt 200 -or $h2.StatusCode -ge 300) { $remoteOk2 = $false }
      } catch {
        $remoteOk2 = $false
      }

      if (-not $remoteOk2) {
        Write-Log ("Pregame signals: remote health check failed ({0}); skipping" -f $remote2)
      } else {
        # If we pushed new artifacts, wait briefly for Render to deploy the new git SHA before calling /api/cards.
        try {
          $doWait = $FinalGitPushSucceeded
          if ($doWait) {
            $waitMax = 300
            try {
              $w = $env:DAILY_PREGAME_PROP_SIGNALS_WAIT_REMOTE_MAX_SEC
              if ($null -ne $w -and $w -ne '') { $waitMax = [int]$w }
            } catch { $waitMax = 300 }
            if ($waitMax -lt 0) { $waitMax = 0 }
            if ($waitMax -gt 1800) { $waitMax = 1800 }

            if ($waitMax -gt 0) {
              $localSha = ''
              try { $localSha = ((& git rev-parse HEAD 2>$null) | Select-Object -First 1).Trim() } catch { $localSha = '' }
              if ($localSha) {
                $deadline = (Get-Date).AddSeconds($waitMax)
                $matched = $false
                while ((Get-Date) -lt $deadline) {
                  try {
                    $ver = Invoke-RestMethod -Uri ("{0}/api/version" -f $remote2) -TimeoutSec 10 -ErrorAction Stop
                    $remoteSha = ''
                    try { $remoteSha = [string]$ver.sha } catch { $remoteSha = '' }
                    if ($remoteSha -and $remoteSha.Trim() -eq $localSha) { $matched = $true; break }
                  } catch { }
                  Start-Sleep -Seconds 20
                }
                if (-not $matched) {
                  Write-Log ("Pregame signals: remote deploy SHA did not match local HEAD after {0}s; skipping emission to avoid stale cards" -f $waitMax)
                  return
                }
                Write-Log 'Pregame signals: remote deploy SHA matched local HEAD'
              }
            }
          }
        } catch {
          Write-Log ("Pregame signals: remote deploy wait failed (continuing): {0}" -f $_.Exception.Message)
        }

        $emitTimeout = 120
        try {
          $t = $env:DAILY_PREGAME_PROP_SIGNAL_TIMEOUT_SEC
          if ($null -ne $t -and $t -ne '') { $emitTimeout = [int]$t }
        } catch { $emitTimeout = 120 }
        if ($emitTimeout -lt 30) { $emitTimeout = 30 }
        if ($emitTimeout -gt 600) { $emitTimeout = 600 }

        Write-Log ("Pregame signals: logging pregame player_prop BETs for {0} (remote={1})" -f $Date, $remote2)
        $rcEmit = Invoke-PyModWithTimeout -plist @(
          'tools/log_pregame_prop_signals.py',
          '--base-url', $remote2,
          '--date', $Date
        ) -TimeoutSeconds $emitTimeout -Label 'pregame_prop_signals'
        $rcEmitText = Format-ExitCodeForLog -Value $rcEmit
        Write-Log ("log_pregame_prop_signals exit code: {0}" -f $rcEmitText)
      }
    } else {
      Write-Log 'Pregame signals: remote base URL missing; skipping'
    }
  } else {
    Write-Log 'Skipping pregame prop signal emission (DAILY_LOG_PREGAME_PROP_SIGNALS=0)'
  }
} catch {
  Write-Log ("Pregame signals failed (non-fatal): {0}" -f $_.Exception.Message)
}
