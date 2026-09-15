@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_windows.ps1" -Source tum -Dataset "%~dp0..\..\work\public_rgbd\rgbd_dataset_freiburg1_xyz"
pause
