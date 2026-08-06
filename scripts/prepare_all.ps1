$ErrorActionPreference = "Stop"
& .venv\Scripts\python.exe scripts\download_ess.py --quicklook
& .venv\Scripts\mirage.exe generate-scm --config configs\data\synthetic.yaml
& .venv\Scripts\mirage.exe preprocess --config configs\data\ess.yaml
& .venv\Scripts\mirage.exe preprocess --config configs\data\boiler.yaml --nrows 100000
& .venv\Scripts\python.exe scripts\verify_external.py

