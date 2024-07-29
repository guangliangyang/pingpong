@echo off
REM 启动Python应用程序
start "" python "C:\Users\PingPong\workspace\PingPong\web\app.py"

REM 等待Python应用程序启动
timeout /t 10 /nobreak

REM 启动Ngrok，并将输出重定向到临时文件
start "" cmd /c "ngrok http 5000 > ngrok_output.txt"

REM 等待Ngrok启动并生成URL
timeout /t 10 /nobreak

REM 从ngrok_output.txt中提取公共URL并打开浏览器
for /f "tokens=*" %%i in ('findstr "https://" ngrok_output.txt') do (
    set NGROK_URL=%%i
    start "" %%i
    goto :END
)

:END
REM 删除临时文件
del ngrok_output.txt

REM 结束批处理脚本
exit
