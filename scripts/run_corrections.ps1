# PowerShell version of run_corrections.sh. From the repo root:
#   powershell -ExecutionPolicy Bypass -File scripts\run_corrections.ps1
$ErrorActionPreference = "Stop"
$PY = if ($env:PY) { $env:PY } else { "python" }
$VENV = ".venv-qk261"
function Run($exe, $args_) { & $exe @args_; if ($LASTEXITCODE -ne 0) { throw "failed: $exe $args_" } }
function Step($t) { Write-Host "`n=== $t ===" }

Step "item 3: no-evidence instances"
Run $PY @("scripts/corrections.py", "noevidence")
Run $PY @("scripts/corrections.py", "noevidence-clinical")
Step "item 2.1: in-context cost law"
Run $PY @("scripts/corrections.py", "law-context")
Step "item 2: environment with qiskit 2.6.1, pytket, pgmpy"
if (-not (Test-Path $VENV)) { Run $PY @("-m", "venv", $VENV) }
$VPY = Join-Path $VENV "Scripts\python.exe"
Run $VPY @("-m", "pip", "install", "-q", "qiskit==2.6.1", "qiskit-algorithms==0.4.0", "pytket==2.18.5", "pytket-qiskit==0.79.0", "pgmpy==1.1.2", "numpy", "scipy", "pandas", "pyarrow")
Step "item 2.2-2.3: circuits, V0-V3 in context, qiskit 2.6.1 and pytket sweeps"
Run $VPY @("scripts/w8.py", "circuits")
Run $VPY @("scripts/corrections.py", "variants-context")
Run $VPY @("scripts/corrections.py", "compile-context", "qiskit")
Run $VPY @("scripts/corrections.py", "compile-context", "tket")
Step "item 2.3: compiler sweep, qiskit 2.2.3 (own environment)"
$V223 = ".venv-qk223"
if (-not (Test-Path $V223)) { Run $PY @("-m", "venv", $V223) }
$PY223 = Join-Path $V223 "Scripts\python.exe"
Run $PY223 @("-m", "pip", "install", "-q", "qiskit==2.2.3", "pgmpy==1.1.2", "numpy", "scipy", "pandas", "pyarrow")
Run $PY223 @("scripts/corrections.py", "compile-context", "qiskit")
Run $PY @("scripts/corrections.py", "compare-context")
Step "item 1 and checkpoint"
Run $PY @("scripts/corrections.py", "sweep")
Run $PY @("scripts/corrections.py", "headline")
Write-Host "`ndone: artifacts/corrections/"
