from flask import Flask, render_template, request
import os
import baseball  # baseball.py의 함수를 사용하기 위해 import

# Flask 애플리케이션 초기화
# template_folder와 static_folder를 명시적으로 지정합니다.
app = Flask(__name__, template_folder='templates', static_folder='static')

# 팀 ID, 이름, 로고 경로를 포함하는 딕셔너리
TEAMS = {
    'LG': {'name': 'LG 트윈스', 'logo': 'images/logos/LG.png'},
    'OB': {'name': '두산 베어스', 'logo': 'images/logos/OB.png'},
    'WO': {'name': '키움 히어로즈', 'logo': 'images/logos/WO.png'},
    'SK': {'name': 'SK 와이번스', 'logo': 'images/logos/SK.png'},
    'KT': {'name': 'KT 위즈', 'logo': 'images/logos/KT.png'},
    'HH': {'name': '한화 이글스', 'logo': 'images/logos/HH.png'},
    'SS': {'name': '삼성 라이온즈', 'logo': 'images/logos/SS.png'},
    'LT': {'name': '롯데 자이언츠', 'logo': 'images/logos/LT.png'},
    'NC': {'name': 'NC 다이노스', 'logo': 'images/logos/NC.png'},
    'HT': {'name': '기아 타이거즈', 'logo': 'images/logos/HT.png'}
}

@app.route('/')
def index():
    """대시보드 메인 페이지를 렌더링합니다."""
    # URL 쿼리에서 'team'과 'month' 파라미터 가져오기
    team_id = request.args.get('team')
    month = request.args.get('month')

    # 데이터 로드
    df_merged = baseball.load_data_and_merge('edit_baseball_2019.csv', 'edit_weather_2019.csv')

    # 사용 가능한 월 목록 가져오기 (팀 선택과 무관하게 전체 데이터 기준)
    available_months = sorted(df_merged['GDAY_DS_DATE'].dt.month.unique())

    is_team_specific = False
    df_kpi_source = df_merged.copy() # KPI 계산을 위한 원본 데이터

    # 선택된 팀 이름 설정
    selected_team_info = None
    if team_id and team_id in TEAMS:
        selected_team_info = TEAMS[team_id]
        df_kpi_source = df_merged[df_merged['T_ID'] == team_id]
        is_team_specific = True

    # 월별 필터링 (KPI 계산용 데이터에만 적용)
    if month and month.isdigit():
        df_kpi_source = df_kpi_source[df_kpi_source['GDAY_DS_DATE'].dt.month == int(month)]

    # KPI 데이터와 차트 HTML 생성
    kpi_data = baseball.get_kpi_data(df_kpi_source, is_team_specific)
    # create_charts에는 필터링되지 않은 전체 df와 선택된 월을 전달
    charts = baseball.create_charts(df_merged, team_id, TEAMS, month)

    return render_template('index.html', 
                           kpi=kpi_data, 
                           charts=charts, 
                           team_links=TEAMS,
                           selected_team_id=team_id,
                           selected_team_info=selected_team_info,
                           available_months=available_months,
                           selected_month=month
                           )

@app.route('/charts/<filename>')
def serve_chart(filename):
    """static/charts 폴더의 차트 HTML 파일을 제공합니다."""
    return render_template(f'charts/{filename}')

if __name__ == '__main__':
    # Flask 앱 실행
    # debug=True: 코드 수정 시 자동 새로고침 (개발용)
    app.run(debug=True)