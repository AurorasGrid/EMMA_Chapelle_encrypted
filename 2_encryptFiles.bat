REM This batch file is used to package the sizing app
@ECHO OFF
ECHO Obfuscating the python code, and converting to .exe file ...
ECHO ----------------------------------
ECHO Output from Pyarmor library
ECHO ------
ECHO.

pyarmor obfuscate --recursive --platform windows.x86_64 --platform linux.armv7 EMMA_Chapelle_Moudon_large_BES.py
pyarmor obfuscate --exact --platform windows.x86_64 --platform linux.armv7 Interfaces\Chapelle_Moudon\large_battery_server.py

rmdir "Ageing_data" /s /q >nul 2>&1
rmdir "Classes" /s /q >nul 2>&1
xcopy /s /y /i Interfaces\Chapelle_Moudon\user_data.yml dist\Interfaces\Chapelle_Moudon\
rmdir "Interfaces" /s /q >nul 2>&1
rmdir "utils" /s /q >nul 2>&1
del "EMMA_Chapelle_Moudon_large_BES.py" /q
del "dist\user_inputs_Chapelle_Moudon_large.py" /q


xcopy /s /y /i dist
rmdir "dist" /s /q >nul 2>&1

ECHO.
ECHO ------
ECHO Python files now encrypted

PAUSE