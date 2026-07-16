@echo off
REM Double-click launcher for the Gaussian Pointcloud Toolkit
cd /d "%~dp0"
python Gaussian_pointcloud_toolkit.py
if errorlevel 1 pause
