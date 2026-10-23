$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $ProjectRoot
try {
    if (-not (Test-Path -LiteralPath '.runtime/python/python.exe')) { throw 'Run the reviewed CPU setup first; it supplies the isolated base runtime.' }
    $env:TEMP = Join-Path $ProjectRoot '.runtime'
    $env:TMP = $env:TEMP
    & .runtime/python/python.exe -m scripts.upgrade_gpu
    if ($LASTEXITCODE -ne 0) { throw 'GPU environment preparation failed' }
    & .runtime/gpu/python.exe -m pip install --no-cache-dir --only-binary=:all: --no-deps torch==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { throw 'CUDA package installation failed' }
    & .runtime/gpu/python.exe -m pip install --no-cache-dir --only-binary=:all: --no-deps triton-windows==3.3.1.post21
    if ($LASTEXITCODE -ne 0) { throw 'Triton package installation failed' }
    & .runtime/gpu/python.exe -m scripts.setup_gpu_headers
    if ($LASTEXITCODE -ne 0) { throw 'Python development headers failed' }
    & .runtime/gpu/python.exe -m scripts.acquire_500m
    if ($LASTEXITCODE -ne 0) { throw 'Candidate checkpoint acquisition failed' }
    & .runtime/gpu/python.exe -m scripts.acquire_v3_data
    if ($LASTEXITCODE -ne 0) { throw 'Public dataset preparation failed' }
} finally { Pop-Location }
