param([string]$Source = 'offline', [string]$Dataset = '')
$ErrorActionPreference = 'Stop'
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    $taskPython = Join-Path $PSScriptRoot '..\..\work\disaster_venv\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $taskPython)) {
    Write-Host '먼저 Python 3.10~3.12로 .venv를 만들고 requirements-windows.txt를 설치하세요. README.md 참고.'
    exit 2
}
$taskArguments = @((Join-Path $PSScriptRoot 'run.py'), '--source', $Source)
if ($Dataset) { $taskArguments += @('--dataset', $Dataset) }
& $taskPython @taskArguments
