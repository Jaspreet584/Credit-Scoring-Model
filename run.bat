@echo off
title RiskFlow - Credit Scoring Server
echo Installing/Verifying dependencies...
pip install pandas scikit-learn numpy fastapi uvicorn pydantic python-multipart
echo.
echo Starting FastAPI server...
python app.py
pause
