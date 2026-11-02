from datetime import datetime
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# 1. 페이지 기본 설정 (브라우저 탭 제목과 아이콘)
st.set_page_config(
    page_title="박스오피스 타임머신", page_icon="🎬", layout="wide"
)

# 앱 제목과 설명
st.title("🎬 박스오피스 타임머신")
st.markdown(
    "월과 일을 선택하면, 최근 10년 동안 해마다 그 날짜에 어떤 영화가 박스오피스"
    " 1위를 차지했는지 살펴볼 수 있습니다."
)

# 2. 비밀 금고(secrets)에서 KOBIS API 인증키 불러오기
try:
  API_KEY = st.secrets["KOBIS_KEY"]
except Exception:
  st.error(
      "⚠️ 스트림릿 비밀 금고(secrets.toml)에 'KOBIS_KEY'가 설정되어 있지"
      " 않습니다. 인증키를 먼저 등록해 주세요."
  )
  st.stop()


# 3. KOBIS API에서 특정 날짜(yyyymmdd)의 일별 박스오피스 데이터를 가져오는 함수
# @st.cache_data를 사용해 한 번 불러온 날짜의 결과는 하루(86400초) 동안 기억합니다.
@st.cache_data(ttl=86400)
def get_box_office_data(target_dt_str):
  url = "https://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json"
  params = {"key": API_KEY, "targetDt": target_dt_str}
  try:
    response = requests.get(url, params=params)
    data = response.json()
    return data
  except Exception as e:
    return None


st.divider()

# 4. 날짜 선택 영역 (자주 찾는 날짜 버튼 + 직접 선택)
st.subheader("📅 조회할 날짜 선택")

# 자주 찾는 날짜 버튼 배치 (클릭 시 세션 상태에 월/일 반영)
col_btn1, col_btn2, col_btn3, col_btn4 = st.columns([1, 1, 1, 3])

if "selected_month" not in st.session_state:
  st.session_state.selected_month = 12
if "selected_day" not in st.session_state:
  st.session_state.selected_day = 25

if col_btn1.button("🎄 크리스마스 (12/25)"):
  st.session_state.selected_month = 12
  st.session_state.selected_day = 25
if col_btn2.button("🇰🇷 광복절 (8/15)"):
  st.session_state.selected_month = 8
  st.session_state.selected_day = 15
if col_btn3.button("👶 어린이날 (5/5)"):
  st.session_state.selected_month = 5
  st.session_state.selected_day = 5

# 월, 일을 고를 수 있는 슬라이더 또는 숫자 입력창
col_m, col_d = st.columns(2)
with col_m:
  selected_month = st.selectbox(
      "월 선택",
      list(range(1, 13)),
      index=st.session_state.selected_month - 1,
  )
with col_d:
  # 각 월의 대략적인 최대 일수 고려 (간이 설정)
  selected_day = st.number_input(
      "일 선택", min_value=1, max_value=31, value=st.session_state.selected_day
  )

st.divider()

# 5. 최근 10년 연도 계산 (올해부터 과거 10년)
current_year = datetime.now().year
target_years = list(range(current_year - 9, current_year + 1))  # 최근 10년

# 데이터 수집 리스트
results_data = []
today = datetime.now().date()

# 진행 바 표시
progress_text = "최근 10년 동안의 박스오피스 데이터를 불러오는 중입니다..."
progress_bar = st.progress(0, text=progress_text)

for idx, year in enumerate(target_years):
  # 유효한 날짜인지 확인 (예: 2월 31일 등 오류 방지)
  try:
    target_date = datetime(year, selected_month, selected_day).date()
  except ValueError:
    # 존재하지 않는 날짜(예: 윤년이 아닌 해의 2월 29일 등) 예외 처리
    results_data.append({
        "year": year,
        "date_str": f"{year}{selected_month:02d}{selected_day:02d}",
        "status": "invalid_date",
    })
    progress_bar.progress((idx + 1) / 10)
    continue

  # 올해 날짜가 아직 오지 않았거나 미래인 경우 집계 전 처리
  if target_date > today:
    results_data.append({
        "year": year,
        "date_str": f"{year}{selected_month:02d}{selected_day:02d}",
        "status": "future_date",
    })
    progress_bar.progress((idx + 1) / 10)
    continue

  # API 호출을 위한 8자리 문자열 생성 (yyyymmdd)
  dt_str = target_date.strftime("%Y%m%d")
  api_response = get_box_office_data(dt_str)

  if api_response and "boxOfficeResult" in api_response:
    box_list = api_response["boxOfficeResult"].get("dailyBoxOfficeList", [])
    if box_list:
      # 1위 영화 정보 추출 (rank가 "1"인 항목)
      rank_1 = box_list[0]
      results_data.append({
          "year": year,
          "date_str": dt_str,
          "status": "success",
          "movieNm": rank_1.get("movieNm"),
          "audiCnt": int(rank_1.get("audiCnt", 0)),  # 숫자로 변환
          "openDt": rank_1.get("openDt"),
          "full_list": box_list,  # TOP 10 전체 표를 위해 보관
      })
    else:
      results_data.append({
          "year": year,
          "date_str": dt_str,
          "status": "no_data",
      })
  else:
    results_data.append(
        {"year": year, "date_str": dt_str, "status": "no_data"}
    )

  progress_bar.progress((idx + 1) / 10)

progress_bar.empty()

# 6. 연도별 1위 영화 카드 형태로 나란히 보여주기
st.subheader(
    f"🏆 최근 10년 ({target_years[0]}년~{target_years[-1]}년)"
    f" {selected_month}월 {selected_day}일 1위 영화"
)

# 카드를 여러 열로 배치 (한 줄에 5개씩 2줄 배치)
cols_per_row = 5
for i in range(0, len(results_data), cols_per_row):
  row_items = results_data[i : i + cols_per_row]
  cols = st.columns(cols_per_row)

  for col_idx, item in enumerate(row_items):
    with cols[col_idx]:
      st.markdown(f"### **{item['year']}년**")
      if item["status"] == "success":
        st.markdown(f"🎬 **{item['movieNm']}**")
        st.markdown(f"👥 관객수: **{item['audiCnt']:,}명**")
        st.markdown(f"📅 개봉일: {item['openDt']}")
      elif item["status"] == "future_date":
        st.info("⏳ 집계 전")
      elif item["status"] == "invalid_date":
        st.warning("⚠️ 존재하지 않는 날짜")
      else:
        st.error("❌ 데이터 없음")
  st.markdown("---")

# 7. 연도별 1위 영화 그날 관객수 세로 막대그래프
st.subheader("📈 연도별 1위 영화 관객수 비교")

# 그래프용 데이터프레임 생성 (집계 성공한 데이터만 필터링)
chart_data = [
    {
        "연도": str(item["year"]),
        "영화명": item["movieNm"],
        "관객수": item["audiCnt"],
    }
    for item in results_data
    if item["status"] == "success"
]

if chart_data:
  df_chart = pd.DataFrame(chart_data)
  # Plotly를 이용한 세로 막대그래프
  fig = px.bar(
      df_chart,
      x="연도",
      y="관객수",
      text="관객수",
      color="관객수",
      hover_data=["영화명"],
      title=f"연도별 {selected_month}월 {selected_day}일 1위 영화 관객수 추이",
      labels={"관객수": "그날 관객수 (명)"},
  )
  fig.update_traces(texttemplate="%{text:,}명", textposition="outside")
  fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide")
  st.plotly_chart(fig, use_container_width=True)
else:
  st.warning("시각화할 관객수 데이터가 충분하지 않습니다.")

st.divider()

# 8. 특정 연도를 골라 그 해 그 날짜의 TOP 10 전체 표 보기
st.subheader("📋 특정 연도 TOP 10 박스오피스 상세 조회")

# 데이터가 존재하는 연도들만 셀렉트박스 옵션으로 제공
available_years = [
    item["year"] for item in results_data if item["status"] == "success"
]

if available_years:
  selected_detail_year = st.selectbox(
      "조회할 연도를 선택하세요", available_years, index=len(available_years) - 1
  )

  # 선택한 연도의 전체 목록 가져오기
  selected_item = next(
      item for item in results_data if item["year"] == selected_detail_year
  )
  top_10_list = selected_item["full_list"]

  # 표로 보여주기 위해 데이터 가공 (숫자 문자열을 정수로 변환)
  table_rows = []
  for movie in top_10_list:
    table_rows.append({
        "순위": int(movie.get("rank", 0)),
        "영화명": movie.get("movieNm"),
        "개봉일": movie.get("openDt"),
        "그날 관객수": int(movie.get("audiCnt", 0)),
        "누적 관객수": int(movie.get("audiAcc", 0)),
        "매출액 점유율(%)": float(movie.get("salesShare", 0)),
    })

  df_top10 = pd.DataFrame(table_rows)

  # 보기 좋게 포맷팅 (콤마 추가)
  st.markdown(
      f"**{selected_detail_year}년 {selected_month}월 {selected_day}일** 박스오피스"
      " TOP 10"
  )
  st.dataframe(
      df_top10.style.format({
          "그날 관객수": "{:,}명",
          "누적 관객수": "{:,}명",
          "매출액 점유율(%)": "{}%",
      }),
      use_container_width=True,
      hide_index=True,
  )
else:
  st.info("선택한 날짜에 조회 가능한 TOP 10 데이터가 없습니다.")
