#!/usr/bin/env bash
# All correction steps in the plan's order (item 3, item 2, item 1, checkpoint). Run from the repo root:
#   bash scripts/run_corrections.sh 2>&1 | tee artifacts/corrections/run.log
# Safe to re-run: the compile steps resume where they stopped.
set -euo pipefail
PY=${PY:-python}                       # your usual environment (qiskit 2.2.3)
VENV=.venv-qk261
step() { echo; echo "=== $* ==="; }

step "item 3: no-evidence instances"
$PY scripts/corrections.py noevidence
$PY scripts/corrections.py noevidence-clinical

step "item 2.1: in-context cost law"
$PY scripts/corrections.py law-context

step "item 2: environment with qiskit 2.6.1, pytket, pgmpy"
if [ ! -d "$VENV" ]; then
  $PY -m venv "$VENV"
fi
if [ -x "$VENV/bin/python" ]; then VPY="$VENV/bin/python"; else VPY="$VENV/Scripts/python.exe"; fi
"$VPY" -m pip install -q qiskit==2.6.1 qiskit-algorithms==0.4.0 pytket==2.18.5 pytket-qiskit==0.79.0 pgmpy==1.1.2 numpy scipy pandas pyarrow

step "item 2.2-2.3: circuits, V0-V3 in context, qiskit 2.6.1 and pytket sweeps"
"$VPY" scripts/w8.py circuits
"$VPY" scripts/corrections.py variants-context
"$VPY" scripts/corrections.py compile-context qiskit
"$VPY" scripts/corrections.py compile-context tket
step "item 2.3: compiler sweep, qiskit 2.2.3 (own environment)"
[ -d .venv-qk223 ] || $PY -m venv .venv-qk223
if [ -x .venv-qk223/bin/python ]; then P223=.venv-qk223/bin/python; else P223=.venv-qk223/Scripts/python.exe; fi
"$P223" -m pip install -q qiskit==2.2.3 pgmpy==1.1.2 numpy scipy pandas pyarrow
"$P223" scripts/corrections.py compile-context qiskit
$PY scripts/corrections.py compare-context

step "item 1 and checkpoint"
$PY scripts/corrections.py sweep
$PY scripts/corrections.py headline
echo; echo "done: artifacts/corrections/"
