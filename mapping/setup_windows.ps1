$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
py -3.12 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 설치가 필요합니다.' }
& '.\.venv\Scripts\python.exe' -m pip install -r requirements-windows.txt
if ($LASTEXITCODE -ne 0) { throw 'Windows 전용 가상환경 설치 실패' }
Write-Host '준비 완료. 화면실행.cmd 또는 start_windows.ps1로 실행하세요.'
