@echo off
title MT5 H1 data pull
echo ============================================================
echo  Step 1/2: installing the MetaTrader5 Python package...
echo ============================================================
py -m pip install --quiet --upgrade MetaTrader5
echo.
echo ============================================================
echo  Step 2/2: pulling H1 history (MT5 terminal must be OPEN)...
echo ============================================================
py "%~dp0mt5_pull.py"
echo.
echo ============================================================
echo  FINISHED. Select all text in this window, copy it,
echo  and paste it back to Claude.
echo ============================================================
pause
