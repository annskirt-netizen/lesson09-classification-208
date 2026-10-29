import json
import os
import pandas as pd
import plotly.express as px
import streamlit as st

# 1. 페이지 설정 (브라우저 탭 제목 및 아이콘)
st.set_page_config(
    page_title="뇌졸중 예측 실습실", page_icon="🧠", layout="wide"
)

# 앱 제목
st.title("🧠 뇌졸중 예측 실습실")
st.markdown("뇌졸중 데이터셋을 탐색하고 주요 정보를 확인하는 실습 공간입니다.")

# 2. 데이터 로드 (UTF-8 인코딩)
DATA_URL = (
    "https://raw.githubusercontent.com/annskirt-netizen/data/refs/heads/main/stroke.csv"
)


@st.cache_data
def load_data(url):
  return pd.read_csv(url, encoding="utf-8")


try:
  df = load_data(DATA_URL)
except Exception as e:
  st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
  st.stop()

# 3. 상단 큰 숫자 카드 네 개 (Metric)
total_people = len(df)
num_columns = len(df.columns)
stroke_count = int((df["stroke"] == 1).sum())
stroke_ratio = (stroke_count / total_people) * 100 if total_people > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("전체 사람 수", f"{total_people:,} 명")
col2.metric("열 개수", f"{num_columns} 개")
col3.metric("뇌졸중(1) 환자 수", f"{stroke_count:,} 명")
col4.metric("뇌졸중 발생 비율", f"{stroke_ratio:.2f}%")

st.divider()

# 4. 내_기록 폴더 및 JSON 파일 경로 설정
os.makedirs("내_기록", exist_ok=True)
meanings_file = os.path.join("내_기록", "stroke_meanings.json")
source_file = os.path.join("내_기록", "stroke_source.json")

# 저장된 우리말 뜻 불러오기
saved_meanings = {}
if os.path.exists(meanings_file):
  try:
    with open(meanings_file, "r", encoding="utf-8") as f:
      saved_meanings = json.load(f)
  except Exception:
    pass

# 표 데이터 구성 (열 이름, 우리말 뜻, 값의 종류, 빈 값 개수)
summary_data = []
for col in df.columns:
  col_type = str(df[col].dtype)
  missing_count = int(df[col].isna().sum())
  meaning = saved_meanings.get(col, "")  # 기본값은 빈 칸
  summary_data.append({
      "열 이름": col,
      "우리말 뜻": meaning,
      "값의 종류": col_type,
      "빈 값 개수": missing_count,
  })

summary_df = pd.DataFrame(summary_data)

st.subheader("📋 데이터 열 정보 및 우리말 뜻 입력")
st.write(
    "교재를 참고하여 **우리말 뜻** 칸을 직접 채우고 아래의 **'우리말 뜻"
    " 저장하기'** 버튼을 누르면 새로고침해도 유지됩니다."
)

# st.data_editor를 사용하여 우리말 뜻 수정 가능하도록 구현
edited_summary_df = st.data_editor(
    summary_df,
    column_config={
        "열 이름": st.column_config.TextColumn("열 이름", disabled=True),
        "우리말 뜻": st.column_config.TextColumn("우리말 뜻 (직접 입력)"),
        "값의 종류": st.column_config.TextColumn("값의 종류", disabled=True),
        "빈 값 개수": st.column_config.NumberColumn("빈 값 개수", disabled=True),
    },
    hide_index=True,
    use_container_width=True,
    key="summary_editor",
)

if st.button("우리말 뜻 저장하기"):
  new_meanings = dict(
      zip(edited_summary_df["열 이름"], edited_summary_df["우리말 뜻"])
  )
  with open(meanings_file, "w", encoding="utf-8") as f:
    json.dump(new_meanings, f, ensure_ascii=False, indent=4)
  st.success("우리말 뜻이 '내_기록/stroke_meanings.json'에 저장되었습니다!")

st.divider()

# 5. 데이터 처음 다섯 줄 표시
st.subheader("🔍 데이터 샘플 (처음 5줄)")
st.dataframe(df.head(5), use_container_width=True)

st.divider()

# 6. 맨 아래 데이터 출처 작성 공간
st.subheader("📖 데이터 출처")
saved_source = ""
if os.path.exists(source_file):
  try:
    with open(source_file, "r", encoding="utf-8") as f:
      data_src = json.load(f)
      saved_source = data_src.get("source", "")
  except Exception:
    pass

source_input = st.text_area(
    "교재를 참고하여 데이터 출처 내용을 입력하세요.",
    value=saved_source,
    height=100,
    key="source_text_area",
)

if st.button("데이터 출처 저장하기"):
  with open(source_file, "w", encoding="utf-8") as f:
    json.dump({"source": source_input}, f, ensure_ascii=False, indent=4)
  st.success("데이터 출처가 '내_기록/stroke_source.json'에 저장되었습니다!")
