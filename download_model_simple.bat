@echo off
echo Downloading YOLOv8n model...
echo.

REM Download using PowerShell if available
powershell -Command "Invoke-WebRequest -Uri 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt' -OutFile 'yolov8n.pt'"

if exist yolov8n.pt (
    echo Model downloaded successfully!
    echo File size: 
    for %%F in (yolov8n.pt) do echo %%~zF bytes
    echo.
    echo You can now run: streamlit run room.py
) else (
    echo Download failed. Please check your internet connection.
    echo.
    echo Alternative: Install wget and run:
    echo wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
)

pause
