$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$root = Split-Path -Parent $PSScriptRoot
$dest = Join-Path $root 'data/installers'
New-Item -ItemType Directory -Path $dest -Force | Out-Null
$wheel = Join-Path $dest 'torch-2.8.0+cu128-cp312-cp312-win_amd64.whl'
Invoke-WebRequest -Uri 'https://download.pytorch.org/whl/cu128/torch-2.8.0%2Bcu128-cp312-cp312-win_amd64.whl?download=1' -OutFile "$wheel.partial" -TimeoutSec 1800
$hash = (Get-FileHash -LiteralPath "$wheel.partial" -Algorithm SHA256).Hash.ToLower()
if ($hash -ne '0ad925202387f4e7314302a1b4f8860fa824357f9b1466d7992bf276370ebcff') { throw "Torch wheel hash mismatch: $hash" }
Move-Item -LiteralPath "$wheel.partial" -Destination $wheel
Write-Output "Verified torch wheel: $wheel"
