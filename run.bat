@echo off
echo.
echo [프로젝트 실행 스크립트]
echo.

REM 가상 환경 폴더(venv)가 있는지 확인
if not exist venv (
    echo "venv" 가상 환경을 생성합니다...
    python -m venv venv
)

echo 가상 환경을 활성화하고 라이브러리를 설치합니다...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo 데이터 분석 스크립트(baseball.py)를 실행하여 KPI 데이터를 생성합니다...
python baseball.py

echo.
echo Flask 웹 서버(app.py)를 시작합니다.
echo 웹 브라우저에서 http://127.0.0.1:5000 로 접속하세요.
echo (서버를 종료하려면 Ctrl+C를 누르세요)
echo.
python app.py