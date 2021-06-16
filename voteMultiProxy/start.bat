echo off
:Restart
taskkill /f /im VoteNetwork.exe
color 0a

cls
ECHO.
ECHO.                     Program is Running .....                           
ECHO.  

:call
start "" "VoteNetwork.exe"

:timeout
TIMEOUT 80 /NOBREAK
cls


:end
goto Restart
pause