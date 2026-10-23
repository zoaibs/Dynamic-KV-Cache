$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $ProjectRoot
try {
    New-Item -ItemType Directory -Force -Path '.runtime' | Out-Null
    $env:TEMP = Join-Path $ProjectRoot '.runtime'
    $env:TMP = $env:TEMP
    if (-not (Test-Path -LiteralPath '.runtime/python/python.exe')) {
        $CurrentBytes = (Get-ChildItem -LiteralPath $ProjectRoot -File -Recurse -Force | Measure-Object Length -Sum).Sum
        if ($CurrentBytes -gt 2800000000) { throw 'Insufficient project budget for a fresh runtime; inspect storage first.' }
        Invoke-WebRequest -UseBasicParsing -Uri 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip' -OutFile '.runtime/python.zip' -TimeoutSec 60
        $RuntimeHash = (Get-FileHash -Algorithm SHA256 -LiteralPath '.runtime/python.zip').Hash
        if ($RuntimeHash -ne '009D6BF7E3B2DDCA3D784FA09F90FE54336D5B60F0E0F305C37F400BF83CFD3B') { throw 'Python archive hash mismatch' }
        Expand-Archive -LiteralPath '.runtime/python.zip' -DestinationPath '.runtime/python'
        Set-Content -LiteralPath '.runtime/python/python311._pth' -Value @('python311.zip','.','..\..','import site')
    }
    if (-not (Test-Path -LiteralPath '.runtime/python/Lib/site-packages/pip')) {
        Invoke-WebRequest -UseBasicParsing -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '.runtime/get-pip.py' -TimeoutSec 60
        $BootstrapHash = (Get-FileHash -Algorithm SHA256 -LiteralPath '.runtime/get-pip.py').Hash
        if ($BootstrapHash -ne 'F4448A01E3AB2914C65F4CA59D4CACBA6B2DBDE19A483FBAC46587F8BBAFF424') { throw 'pip bootstrap changed; inspect the official script before updating the pinned hash' }
        & .runtime/python/python.exe .runtime/get-pip.py --no-cache-dir --no-setuptools --no-wheel
        if ($LASTEXITCODE -ne 0) { throw 'pip initialization failed' }
    }
    & .runtime/python/python.exe -m pip install --no-cache-dir --only-binary=:all: setuptools==80.9.0 wheel==0.45.1
    if ($LASTEXITCODE -ne 0) { throw 'Build-tool installation failed' }
    & .runtime/python/python.exe -m pip install --no-cache-dir --no-build-isolation -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Project dependency installation failed' }
    & .runtime/python/python.exe -m scripts.resources
    if ($LASTEXITCODE -ne 0) { throw 'Checkpoint download failed' }
    & .runtime/python/python.exe -m scripts.acquire_data
    if ($LASTEXITCODE -ne 0) { throw 'Dataset preparation failed' }
} finally {
    Pop-Location
}
