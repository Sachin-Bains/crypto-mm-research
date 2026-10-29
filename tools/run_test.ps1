# The out-of-sample test of versions 2 and 3, end to end. Run from the project folder:
#   powershell -ExecutionPolicy Bypass -File tools\run_test.ps1
# Everything printed is also saved to data\results\test_run_log.txt.
# Leave the recorder running in its own window. Expect 10 to 14 hours: start it in the evening.

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force data\results | Out-Null
Start-Transcript -Path data\results\test_run_log.txt -Append
$py = ".venv\Scripts\python"
function Run { & $py @args; if ($LASTEXITCODE -ne 0) { throw "Step failed: $args" } }

$build = "2026-10-20,2026-10-21,2026-10-22,2026-10-23,2026-10-24,2026-10-25,2026-10-26,2026-10-27,2026-10-28"
$all   = "2026-10-14,2026-10-15,2026-10-16,2026-10-17,2026-10-18,$build"
$iv    = "2026-10-14,2026-10-28"

Write-Host "`n=== 1/4 Rebuild books, 20 to 28 October ==="
Run -m rebuild --start 2026-10-20 --end 2026-10-28
Write-Host "`n=== 2/4 Features ==="
Run -m forecast.features --dates $build
Write-Host "`n=== 3/4 Forecasts and intervals (walk-forward, all days) ==="
Run -m forecast.evaluate --dates $all
Run -m forecast.intervals --dates $all

Write-Host "`n=== 4/4 Strategies on the test days ==="
$prev = "2026-10-18_v3dev"
foreach ($d in 22..28) {
    $day = "2026-10-$d"
    Run -m sim.frontier --date $day --intervals-dates $iv --only join,full_pull,random_off_pull,vol_pull,conformal,conformal_q
    Run -m sim.frontier --date $day --intervals-dates $iv --only conformal,conformal_q --lam0-from $prev --label warm
    $prev = "${day}_warm"
}

Write-Host "`n=== Results: pre-registered test days (22 and 23 October) ==="
Run tools\pool_frontier.py --dates 2026-10-22,2026-10-23
Run tools\pool_frontier.py --dates 2026-10-22,2026-10-23 --label warm
Write-Host "`n=== Results: extension (24 to 28 October) ==="
Run tools\pool_frontier.py --dates 2026-10-24,2026-10-25,2026-10-26,2026-10-27,2026-10-28
Run tools\pool_frontier.py --dates 2026-10-24,2026-10-25,2026-10-26,2026-10-27,2026-10-28 --label warm
Stop-Transcript
