# FXB — forex signal bot (foundation + data layer)

## Setup (PowerShell 5.1)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env   # then edit values
```

## Run tests (offline, no broker required)
```powershell
pytest -q
```
