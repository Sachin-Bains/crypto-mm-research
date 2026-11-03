# Avellaneda-Stoikov benchmark and replay export. Run from the project folder:
#   powershell -ExecutionPolicy Bypass -File tools\run_next.ps1
# Everything printed is saved to data\results\run_next_log.txt. Expect about 3 to 4 hours.
# Leave the recorder running in its own window.

$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
New-Item -ItemType Directory -Force data\results | Out-Null
$log = "data\results\run_next_log.txt"
$py = ".venv\Scripts\python"
function Run {
    "`n> python $args" | Tee-Object -FilePath $log -Append
    & $py @args 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { "STEP FAILED: $args" | Tee-Object -FilePath $log -Append; exit 1 }
}

$test = "2026-10-22,2026-10-23,2026-10-24,2026-10-25,2026-10-26,2026-10-27,2026-10-28"
$iv   = "2026-10-14,2026-10-28"

"=== 1/4 Liquidations and clock samples for the test days ===" | Tee-Object -FilePath $log -Append
Run -m research.events --dates $test
"=== 2/4 Avellaneda-Stoikov calibration, development days only ===" | Tee-Object -FilePath $log -Append
Run tools\calibrate_as.py --dates 2026-10-15,2026-10-16,2026-10-17,2026-10-18
"=== 3/4 Avellaneda-Stoikov on the test days ===" | Tee-Object -FilePath $log -Append
foreach ($d in 22..28) {
    Run -m sim.frontier --date "2026-10-$d" --intervals-dates $iv --only "as,as_skew" --label as
}
"=== 4/4 Replay windows ===" | Tee-Object -FilePath $log -Append
Run tools\export_replay.py --dates $test --intervals-dates $iv --auto
"=== Done ===" | Tee-Object -FilePath $log -Append
