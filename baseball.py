import pandas as pd
import numpy as np
import plotly.express as px
import os
from scipy.stats import linregress
import plotly.graph_objects as go
from io import StringIO
from plotly.subplots import make_subplots


# Flask 프로젝트의 루트 디렉토리를 기준으로 static 폴더 설정
STATIC_DIR = "static" 
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

# 2019 KBO 홈구장 기준 팀-도시 매핑
team_city_map = {
    'OB': '서울', 'LG': '서울', 'WO': '서울', 'KT': '수원', 'SK': '인천',
    'HH': '대전', 'SS': '대구', 'LT': '부산', 'NC': '창원', 'HT': '광주'
}

def load_data_and_merge(baseball_file, weather_file):
    """
    데이터를 로드하고 전처리하여 병합된 DataFrame을 반환합니다.
    """
    
    # 1. 데이터 로드 (utf-8 인코딩 사용)
    df_baseball = pd.read_csv(baseball_file, encoding='utf-8')
    df_weather = pd.read_csv(weather_file, encoding='cp949')

    # 날씨 데이터의 결측치를 0으로 채움
    df_weather.fillna(0, inplace=True)


    # 2. baseball 데이터 전처리
    df_baseball['GDAY_DS_DATE'] = pd.to_datetime(df_baseball['GDAY_DS'], format='%Y%m%d')
    df_baseball['GDAY_DS_STR'] = df_baseball['GDAY_DS'].astype(str)

    # 3. weather 데이터 전처리: 일자별 최대/평균 계산
    df_weather['일시_DATETIME'] = pd.to_datetime(df_weather['일시'], format='%Y-%m-%d %H:%M')
    df_weather['GDAY_DS_STR'] = df_weather['일시_DATETIME'].dt.strftime('%Y%m%d')

    df_weather_daily = df_weather.groupby(['지점명', 'GDAY_DS_STR']).agg(
        Avg_Temp=('기온(°C)', 'mean'),
        Total_Rain=('강수량(mm)', 'sum'),
        Max_WindSpeed=('풍속(m/s)', 'max'),
        Avg_Humidity=('습도(%)', 'mean') # 습도 데이터 추가
    ).reset_index()
    df_weather_daily.rename(columns={'지점명': 'CITY', 'GDAY_DS_STR': 'GDAY_DS_STR'}, inplace=True)
    
    df_weather_daily['Is_Rain'] = np.where(df_weather_daily['Total_Rain'] > 0, 1, 0)

    # 4. 팀-도시 매핑 및 병합
    df_city_map = pd.DataFrame(list(team_city_map.items()), columns=['T_ID', 'CITY'])
    
    df_merged = pd.merge(df_baseball, df_city_map, on='T_ID', how='left')
    
    df_merged = pd.merge(
        df_merged, 
        df_weather_daily, 
        on=['GDAY_DS_STR', 'CITY'], 
        how='left'
    )
    
    df_merged.rename(columns={'RUN': 'RUN'}, inplace=True)
    
    # df_merged.dropna(subset=['Avg_Temp', 'Total_Rain', 'Max_WindSpeed'], inplace=True)
    # df_merged.dropna(subset=['Avg_Temp', 'Total_Rain', 'Max_WindSpeed', 'Avg_Humidity'], inplace=True)

    return df_merged

def run_analysis():
    print("1. 데이터 로드 및 전처리 시작...")
    
    df_merged = load_data_and_merge('edit_baseball_2019.csv', 'edit_weather_2019.csv')
    
    # --- 2. KPI 계산 ---
    total_games = int(len(df_merged) / 2) 
def get_kpi_data(df_merged, is_team_specific):
    """DataFrame을 기반으로 KPI 데이터를 계산합니다."""
    if df_merged.empty:
        return {'total_games': 0, 'avg_temp': 'N/A', 'total_hr': 0}

    # 전체 분석일 경우 경기 수는 2로 나누고, 팀별 분석은 df 길이를 그대로 사용
    total_games = int(len(df_merged) / 2) if not is_team_specific else len(df_merged)
    avg_temp = round(df_merged['Avg_Temp'].mean(), 1)
    # 전체 분석일 경우 홈런 수는 2로 나누고, 팀별 분석은 합계를 그대로 사용
    total_hr = int(df_merged['HR'].sum() / 2) if not is_team_specific else int(df_merged['HR'].sum())

    kpi_content = f"total_games:{total_games}\navg_temp:{avg_temp}\ntotal_hr:{total_hr}\n"
    with open('kpi_data.txt', "w", encoding="utf-8") as f:
        f.write(kpi_content)
    print("2. KPI 데이터 생성 완료: kpi_data.txt")
    
    # --- 3. 기존 시각화 파일 생성 (Placeholder) ---
    # (실제 분석 로직은 생략하고 파일만 생성하여 앱이 오류 없이 실행되도록 합니다.)
    return {'total_games': total_games, 'avg_temp': f"{avg_temp}°C", 'total_hr': total_hr}

def create_strength_comparison_charts(df_merged):
    """
    강팀과 약팀의 습도/온도에 따른 타율 및 홈런을 비교하는 차트를 생성합니다.
    """
    # 1. 팀 순위 계산
    team_stats = df_merged.groupby('T_ID').agg(
        Games=('win', 'size'),
        Wins=('win', 'sum')
    ).reset_index()
    team_stats['WinRate'] = (team_stats['Wins'] / team_stats['Games'])
    team_stats = team_stats.sort_values(by='WinRate', ascending=False)

    # 2. 강팀(상위 5)과 약팀(하위 5) 분류
    strong_teams = team_stats.head(5)['T_ID'].tolist()
    weak_teams = team_stats.tail(5)['T_ID'].tolist()

    df_strength = df_merged[df_merged['T_ID'].isin(strong_teams + weak_teams)].copy()
    
    def get_team_group(t_id):
        if t_id in strong_teams:
            return '강팀'
        elif t_id in weak_teams:
            return '약팀'
        return None
    
    df_strength['Team_Group'] = df_strength['T_ID'].apply(get_team_group)

    df_strength['AVG'] = (df_strength['HIT'] / df_strength['AB']).fillna(0)

    # 3. 타율 비교 산점도 생성
    fig_avg = px.scatter(
        df_strength, x="Avg_Humidity", y="Avg_Temp", 
        color="Team_Group",
        size='AVG',
        hover_data=['T_ID', 'AVG'],
        labels={'Avg_Humidity': '평균 습도(%)', 'Avg_Temp': '평균 온도(°C)', 'AVG': '타율', 'Team_Group': '팀 그룹'},
        title="<b>강팀 vs 약팀: 습도/온도에 따른 타율 비교</b>"
    )

    # 4. 홈런 비교 산점도 생성
    fig_hr = px.scatter(
        df_strength, x="Avg_Humidity", y="Avg_Temp",
        color="Team_Group",
        size='HR',
        hover_data=['T_ID', 'HR'],
        labels={'Avg_Humidity': '평균 습도(%)', 'Avg_Temp': '평균 온도(°C)', 'HR': '홈런', 'Team_Group': '팀 그룹'},
        title="<b>강팀 vs 약팀: 습도/온도에 따른 홈런 비교</b>"
    )

    return fig_avg, fig_hr

def create_monthly_dual_axis_chart(df_merged, team_id):
    """
    팀별 월별 습도/온도 및 타율/홈런을 이중 축 선 그래프로 표시합니다.
    """
    df_team = df_merged[df_merged['T_ID'] == team_id].copy()
    df_team['AVG'] = (df_team['HIT'] / df_team['AB']).fillna(0)
    df_team['Month'] = df_team['GDAY_DS_DATE'].dt.month
    
    # 월별 평균 습도, 온도, 타율, 홈런 계산
    monthly_data = df_team.groupby('Month').agg(
        Avg_Humidity=('Avg_Humidity', 'mean'),
        Avg_Temp=('Avg_Temp', 'mean'),
        Avg_WindSpeed=('Max_WindSpeed', 'mean'),  # 월별 평균 풍속 추가
        AVG=('AVG', 'mean'),
        HR=('HR', 'sum') # 홈런은 합계로
    ).reset_index()
    
    # 이중 축 그래프 생성
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 습도 및 온도 (왼쪽 축)
    fig.add_trace(
        go.Scatter(x=monthly_data['Month'], y=monthly_data['Avg_Humidity'], name="평균 습도(%)", marker_color='blue', line=dict(dash='dot')),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=monthly_data['Month'], y=monthly_data['Avg_Temp'], name="평균 온도(°C)", marker_color='red', line=dict(dash='dot')),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=monthly_data['Month'], y=monthly_data['Avg_WindSpeed'], name="평균 풍속(m/s)", marker_color='green', line=dict(dash='dot')),
        secondary_y=False,
    )
    
    # 타율 및 홈런 (오른쪽 축)
    fig.add_trace(
        go.Scatter(x=monthly_data['Month'], y=monthly_data['AVG'], name="평균 타율", marker_color='green'),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(x=monthly_data['Month'], y=monthly_data['HR'], name="홈런 수", marker_color='purple'),
        secondary_y=True,
    )
    
    # 축 레이블 및 제목 업데이트
    fig.update_layout(
        title_text=f"<b>{team_id} 팀: 월별 습도/온도 및 타율/홈런 변화</b>",
        title_x=0.5,
        xaxis=dict(title='월', fixedrange=True),
        yaxis=dict(title=None, fixedrange=True),  # 왼쪽 y축 제목 숨기기
        yaxis2=dict(title=None, fixedrange=True), # 오른쪽 y축 제목 숨기기
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), # 범례 위치 조정
        margin=dict(l=100, r=50, t=80, b=50) # 그래프 여백 조정 (왼쪽 l을 늘려 오른쪽으로 이동)
    )

    # 주석을 사용하여 가로 방향의 y축 제목 추가
    fig.add_annotation(
        x=-0.09, y=0.45, xref='paper', yref='paper',
        text="습도/온도/풍속", showarrow=False,
        font=dict(size=12),
        xanchor='left', yanchor='bottom'
    )
    fig.add_annotation(
        x=1.02, y=0.45, xref='paper', yref='paper',
        text="타율/홈런", showarrow=False,
        font=dict(size=12),
        xanchor='right', yanchor='bottom'
    )
    
    return fig

def create_correlation_chart(df_team, team_name, month=None):
    """
    팀의 주요 경기력 지표와 승패의 상관관계를 보여주는 바 차트를 생성합니다.
    month 파라미터로 월별 필터링이 가능합니다.
    """
    df_corr = df_team.copy()
    period_text = "전체 기간"

    # 월별 필터링
    if month and month.isdigit():
        df_corr = df_corr[df_corr['GDAY_DS_DATE'].dt.month == int(month)]
        period_text = f"{month}월"

    # 지표 이름 한글 매핑 (이 위치로 이동)
    indicator_map = {
        'RUN': '득점',
        'HIT': '안타',
        'HR': '홈런',
        'RBI': '타점',
        'OBP': '출루율',
        'KK': '삼진'
    }

    # 상관관계 분석을 위한 지표 선택
    # OBP, RBI, KK 컬럼이 없으므로, 존재하는 컬럼으로 대체하거나 주석처리합니다.
    # 사용 가능한 지표: 'RUN', 'HIT', 'HR'
    # 상관관계 계산을 위해 숫자형 데이터만 선택
    corr_cols = ['win', 'RUN', 'HIT', 'HR'] # 기본 지표
    
    # 'OBP', 'RBI', 'KK' 컬럼이 df_team에 있는지 확인 후 추가
    if 'RBI' in df_corr.columns:
        corr_cols.append('RBI')
    if 'OBP' in df_corr.columns:
        corr_cols.append('OBP')
    if 'KK' in df_corr.columns:
        corr_cols.append('KK')

    correlations = df_corr[corr_cols].corr()['win'].drop('win').sort_values(ascending=True)

    # 인덱스(지표 이름)를 한글로 변경
    correlations.index = correlations.index.map(lambda x: indicator_map.get(x, x)) # indicator_map을 사용하도록 수정

    # 차트 생성
    fig = go.Figure(go.Bar(
        x=correlations.values,
        y=correlations.index, # 한글로 변경된 인덱스 사용
        orientation='h',
        marker_color=['#d62728' if c < 0 else '#1f77b4' for c in correlations.values],
        text=[f'{c:.2f}' for c in correlations.values],
        textposition='auto'
    ))

    # 차트 제목에 색상 범례 추가
    title_with_legend = f'<b>{team_name} ({period_text}): 경기력 지표별 승패 상관관계</b><br><span style="font-size: 12px;">(<span style="color:#1f77b4;">■</span> 양의 상관관계, <span style="color:#d62728;">■</span> 음의 상관관계)</span>'

    fig.update_layout(
        title_text=title_with_legend,
        title_x=0.5,
        xaxis=dict(title="상관계수 (vs 승리)", fixedrange=True), # X축 확대/축소 비활성화
    )
    return fig

def create_charts(df_merged, team_id=None, teams_info=None, month=None):
    """DataFrame을 기반으로 모든 차트를 생성하고 HTML 문자열 딕셔너리를 반환합니다."""
    # config: Plotly 차트의 상단 메뉴(modebar)를 숨깁니다.
    html_config = {
        'full_html': False,
        'include_plotlyjs': 'cdn',
        'config': {'displayModeBar': False}
    }   
    fig = None  # fig 변수를 초기화합니다.

    if team_id: # 특정 팀이 선택된 경우
        df_team_base = df_merged[df_merged['T_ID'] == team_id].copy()
        team_name = teams_info[team_id]['name'] if teams_info and team_id in teams_info else team_id

        # 월별 필터링 적용
        df_team = df_team_base
        period_text = "전체"
        if month and month.isdigit():
            df_team = df_team_base[df_team_base['GDAY_DS_DATE'].dt.month == int(month)]
            period_text = f"{month}월"

        # --- 강수 유무별 승률 분석 ---
        df_no_rain = df_team[df_team['Is_Rain'] == 0]
        games_no_rain = len(df_no_rain)
        wins_no_rain = int(df_no_rain['win'].sum())
        losses_no_rain = games_no_rain - wins_no_rain
        win_rate_no_rain = wins_no_rain / games_no_rain if games_no_rain > 0 else 0

        df_rain = df_team[df_team['Is_Rain'] == 1]
        games_rain = len(df_rain)
        wins_rain = int(df_rain['win'].sum())
        losses_rain = games_rain - wins_rain
        win_rate_rain = wins_rain / games_rain if games_rain > 0 else 0
        
        # 차트 생성을 위한 데이터 준비
        conditions = ['비 안 온 날', '비 온 날']
        win_rates = [win_rate_no_rain, win_rate_rain]
        wins = [wins_no_rain, wins_rain]
        losses = [losses_no_rain, losses_rain]
        colors = ['#636EFA', '#EF553B']
        
        # 메인 차트: 100% 누적 막대 차트
        fig = go.Figure() # 단일 차트이므로 make_subplots 대신 Figure 사용

        # 100% 누적 막대 차트 (승/패 비율)
        # row, col 인자를 제거합니다. 단일 차트에는 필요하지 않습니다.
        fig.add_trace(go.Bar(name='승리', y=conditions, x=wins, orientation='h', marker_color='#1f77b4',
                             text=[f'{wr:.1%}' for wr in win_rates], textposition='inside',
                             hovertemplate='<b>%{y}</b><br>승리: %{x}경기<br>승률: %{text}<extra></extra>'))
        fig.add_trace(go.Bar(name='패배', y=conditions, x=losses, orientation='h', marker_color='#d62728',
                             text=[f'{1-wr:.1%}' for wr in win_rates], textposition='inside',
                             hovertemplate='<b>%{y}</b><br>패배: %{x}경기<br>패배율: %{text}<extra></extra>'))

        # 레이아웃 업데이트
        fig.update_layout(
            barmode='stack',
            yaxis=dict(autorange='reversed', fixedrange=True),
            xaxis=dict(title='경기 수', fixedrange=True),
            showlegend=False
        )

        # 차트 제목 설정
        fig.update_layout(title_text=f'<b>{team_name} 강수 유무별 승률 분석 ({period_text})</b>', title_x=0.5)

        fig_monthly = create_monthly_dual_axis_chart(df_merged, team_id)
        fig_corr = create_correlation_chart(df_team_base, team_name, month)

        # --- 상대 전적 분석 (강한 팀/약한 팀 TOP 3) ---
        # 월별 필터가 있으면 해당 월의 데이터만 사용, 없으면 전체 데이터 사용
        df_head_to_head_source = df_team_base
        if month and month.isdigit():
            df_head_to_head_source = df_team_base[df_team_base['GDAY_DS_DATE'].dt.month == int(month)]

        head_to_head = df_head_to_head_source.groupby('VS_T_ID').agg(
            games=('win', 'size'),
            wins=('win', 'sum')
        ).reset_index()
        head_to_head['win_rate'] = (head_to_head['wins'] / head_to_head['games']).fillna(0)
        head_to_head_sorted = head_to_head.sort_values(by='win_rate', ascending=False)

        strong_opponents_df = head_to_head_sorted.head(3)
        weak_opponents_df = head_to_head_sorted.tail(3).sort_values(by='win_rate', ascending=True)

        def format_opponent_data(df, teams_info):
            data = []
            for _, row in df.iterrows():
                opponent_id = row['VS_T_ID']
                opponent_info = teams_info.get(opponent_id, {})
                data.append({
                    'id': opponent_id,
                    'name': opponent_info.get('name', opponent_id),
                    'logo': opponent_info.get('logo', ''),
                    'win_rate': f"{row['win_rate']:.3f}",
                    'record': f"{int(row['wins'])}승 {int(row['games'] - row['wins'])}패"
                })
            return data

        strong_opponents_data = format_opponent_data(strong_opponents_df, teams_info)
        weak_opponents_data = format_opponent_data(weak_opponents_df, teams_info)

        return {
            'main_chart': fig.to_html(**html_config),
            'monthly_chart': fig_monthly.to_html(**html_config),
            'correlation_chart': fig_corr.to_html(**html_config),
            'strong_opponents': strong_opponents_data,
            'weak_opponents': weak_opponents_data
        }

    else: # 전체 팀 보기
        # 1x2 subplot 생성 (왼쪽: 도넛 차트, 오른쪽: 순위 테이블)
        fig = make_subplots( # fig 변수를 여기서 새로 정의
            rows=1, cols=2,
            column_widths=[0.4, 0.6],
            specs=[[{'type': 'domain'}, {'type': 'table'}]],
            subplot_titles=("<b>강수 유무별 경기 비율</b>", "<b>2019 시즌 팀 순위</b>")
        )

        # --- 왼쪽: 강수 유무에 따른 경기 수 (도넛 차트) ---
        rain_counts = df_merged['Is_Rain'].value_counts()
        rain_counts = rain_counts / 2 # 각 경기는 두 팀에 대해 한 번씩 나타나므로 2로 나눔

        df_rain_games = pd.DataFrame({
            'label': ['비 안 온 날', '비 온 날'],
            'games': [rain_counts.get(0, 0), rain_counts.get(1, 0)]
        })

        fig.add_trace(go.Pie(
            labels=df_rain_games['label'],
            values=df_rain_games['games'],
            hole=.4,
            marker_colors=['#636EFA', '#EF553B'],
            textinfo='percent+label+value',
            hoverinfo='label+percent+value'
        ), row=1, col=1)

        # --- 오른쪽: 팀 순위 (테이블) ---
        team_stats = df_merged.groupby('T_ID').agg(
            Games=('win', 'size'),
            Wins=('win', 'sum')
        ).reset_index()
        team_stats['Losses'] = team_stats['Games'] - team_stats['Wins']
        team_stats['WinRate'] = (team_stats['Wins'] / team_stats['Games']).round(3)
        team_stats = team_stats.sort_values(by='WinRate', ascending=False)
        team_stats['Rank'] = range(1, len(team_stats) + 1)

        # 템플릿에서 사용할 팀 순위 데이터 생성
        team_rankings_data = []
        for _, row in team_stats.iterrows():
            team_id = row['T_ID']
            team_info = teams_info.get(team_id, {})
            team_rankings_data.append({
                'rank': row['Rank'], 'id': team_id, 'name': team_info.get('name', team_id),
                'logo': team_info.get('logo', ''), 'win_rate': f"{row['WinRate']:.3f}"
            })
        team_stats['Rank'] = range(1, len(team_stats) + 1)
        
        # 테이블에 표시할 열 선택 및 이름 변경
        team_stats_display = team_stats[['Rank', 'T_ID']]
        # 헤더 이름 변경
        header_values = ['순위', '팀']

        fig.add_trace(go.Table(
            header=dict(values=header_values,
                        fill_color='paleturquoise',
                        align='center'),
            cells=dict(values=[team_stats_display.Rank, team_stats_display.T_ID],
                       fill_color='lavender', 
                       align='center')
        ), row=1, col=2)
        fig.update_layout(title_text='<b>KBO 리그 대시보드</b>', title_x=0.5, showlegend=False)
        
        return {
            'main_chart': fig.to_html(**html_config),
            'team_rankings': team_rankings_data
        }


if __name__ == '__main__':
    # 이 스크립트를 직접 실행할 때의 동작 (예: 데이터 분석 및 차트 생성 테스트)
    print("1. 데이터 로드 및 전처리 시작...")
    # 테스트를 위한 TEAMS 정보 (app.py와 동일하게 정의)
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
    df_merged = load_data_and_merge('edit_baseball_2019.csv', 'edit_weather_2019.csv')
    print("데이터 로드 완료.")
    
    print("\n2. 전체 KPI 데이터 계산...")
    kpis = get_kpi_data(df_merged, is_team_specific=False)
    print(kpis)
    
    print("\n3. 전체 차트 생성 (HTML 파일로 저장)...")
    if not os.path.exists(STATIC_DIR):
        os.makedirs(STATIC_DIR)
    
    charts_html = create_charts(df_merged, team_id=None, teams_info=TEAMS)
    print("\n3. 전체 차트 생성 완료.")