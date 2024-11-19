@ECHO OFF
@SET PYTHONIOENCODING=utf-8
@SET PYTHONUTF8=1
@FOR /F "tokens=2 delims=:." %%A in ('chcp') do for %%B in (%%A) do set "_CONDA_OLD_CHCP=%%B"
@chcp 65001 > NUL
@CALL "C:\Users\ibrah\miniforge3\condabin\conda.bat" activate "c:\Users\ibrah\computer_vision\comp0241_24\.conda"
@IF %ERRORLEVEL% NEQ 0 EXIT /b %ERRORLEVEL%
@c:\Users\ibrah\computer_vision\comp0241_24\.conda\python.exe -Wi -m compileall -q -l -i C:\Users\ibrah\AppData\Local\Temp\tmpxmaaorvh -j 0
@IF %ERRORLEVEL% NEQ 0 EXIT /b %ERRORLEVEL%
@chcp %_CONDA_OLD_CHCP%>NUL
