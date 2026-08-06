$ErrorActionPreference = "Stop"
& .venv\Scripts\mirage.exe generate-scm --config configs\data\synthetic_smoke.yaml
if (Test-Path -LiteralPath "data\external\ess\quicklook\preprocessedData.csv") {
    & .venv\Scripts\mirage.exe preprocess --config configs\data\ess.yaml
}
if (Test-Path -LiteralPath "data\raw\boiler\sample\数据样例.csv") {
    & .venv\Scripts\mirage.exe preprocess --config configs\data\boiler.yaml --nrows 5000 --output data\processed\boiler\sample_5000
}
& .venv\Scripts\mirage.exe train --config configs\experiment\smoke.yaml
& .venv\Scripts\mirage.exe evaluate --run-dir artifacts\smoke
& .venv\Scripts\python.exe scripts\verify_external.py
& .venv\Scripts\python.exe -m pytest -q
