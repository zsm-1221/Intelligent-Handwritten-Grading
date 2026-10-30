$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
& '.\.venv\Scripts\python.exe' -X utf8 demo/app.py
