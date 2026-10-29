import pandas as pd
import plotly.express as px
import streamlit as st

# 페이지 설정
st.set_page_config(
    page_title="데이터 탐색 - 뇌졸중 예측 실습실", page_icon="📊", layout="wide"
)

st.title("📊 뇌졸중 데이터 탐색")
st.markdown(
    "데이터의 분포와 주요 변수 간의 관계를 시각화와 표를 통해 탐색하는"
    " 공간입니다."
)

# 데이터 로드 (첫 화면과 동일한 주소 및 UTF-8 인코딩)
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

st.divider()

# 1. 나이와 평균 혈당의 분포를 히스토그램 두 개로 나란히
st.subheader("1. 나이와 평균 혈당 분포")
col1, col2 = st.columns(2)

with col1:
  fig_age = px.histogram(
      df, x="age", nbins=30, title="나이(age) 분포", labels={"age": "나이"}
  )
  st.plotly_chart(fig_age, use_container_width=True)

with col2:
  fig_glucose = px.histogram(
      df,
      x="avg_glucose_level",
      nbins=30,
      title="평균 혈당(avg_glucose_level) 분포",
      labels={"avg_glucose_level": "평균 혈당"},
  )
  st.plotly_chart(fig_glucose, use_container_width=True)

st.divider()

# 2. 뇌졸중 여부에 따른 나이와 평균 혈당 상자그림 및 평균값 표
st.subheader("2. 뇌졸중 여부에 따른 나이 및 평균 혈당 비교")
# stroke 컬럼을 보기 편하게 문자열로 변환 (0 -> 정상, 1 -> 뇌졸중)
df["stroke_label"] = df["stroke"].map({0: "정상 (0)", 1: "뇌졸중 (1)"})

col3, col4 = st.columns(2)

with col3:
  fig_box_age = px.box(
      df,
      x="stroke_label",
      y="age",
      color="stroke_label",
      title="뇌졸중 여부에 따른 나이 비교",
      labels={"stroke_label": "뇌졸중 여부", "age": "나이"},
  )
  st.plotly_chart(fig_box_age, use_container_width=True)

with col4:
  fig_box_glucose = px.box(
      df,
      x="stroke_label",
      y="avg_glucose_level",
      color="stroke_label",
      title="뇌졸중 여부에 따른 평균 혈당 비교",
      labels={
          "stroke_label": "뇌졸중 여부",
          "avg_glucose_level": "평균 혈당",
      },
  )
  st.plotly_chart(fig_box_glucose, use_container_width=True)

# 두 그룹의 평균값 표
st.write("**[두 그룹의 평균값 비교]**")
mean_table = (
    df.groupby("stroke_label")[["age", "avg_glucose_level"]]
    .mean()
    .rename(
        columns={
            "age": "평균 나이",
            "avg_glucose_level": "평균 혈당(avg_glucose_level)",
        }
    )
    .reset_index()
)
st.dataframe(mean_table, use_container_width=True, hide_index=True)

st.divider()

# 3. 고혈압 및 심장병 여부에 따른 뇌졸중 비율 막대그래프
st.subheader("3. 기질적 질환(고혈압, 심장병)과 뇌졸중 비율")


# 비율 계산 함수
def get_rate_df(col_name, label_map):
  temp_df = df.copy()
  temp_df[col_name] = temp_df[col_name].map(label_map)
  grouped = (
      temp_df.groupby(col_name)["stroke"]
      .agg(환자수="sum", 전체인원="count")
      .reset_index()
  )
  grouped["뇌졸중 비율(%)"] = (grouped["환자수"] / grouped["전체인원"]) * 100
  return grouped


col5, col6 = st.columns(2)

with col5:
  hypertension_df = get_rate_df(
      "hypertension", {0: "고혈압 없음 (0)", 1: "고혈압 있음 (1)"}
  )
  fig_hyp = px.bar(
      hypertension_df,
      x="hypertension",
      y="뇌졸중 비율(%)",
      color="hypertension",
      title="고혈압 여부에 따른 뇌졸중 비율",
      text_auto=".2f",
      labels={"hypertension": "고혈압 여부"},
  )
  st.plotly_chart(fig_hyp, use_container_width=True)

with col6:
  heart_df = get_rate_df(
      "heart_disease", {0: "심장병 없음 (0)", 1: "심장병 있음 (1)"}
  )
  fig_heart = px.bar(
      heart_df,
      x="heart_disease",
      y="뇌졸중 비율(%)",
      color="heart_disease",
      title="심장병 여부에 따른 뇌졸중 비율",
      text_auto=".2f",
      labels={"heart_disease": "심장병 여부"},
  )
  st.plotly_chart(fig_heart, use_container_width=True)

st.divider()

# 4. BMI가 비어 있는 사람들의 뇌졸중 비율과 전체 뇌졸중 비율 비교
st.subheader("4. 체질량지수(bmi) 결측치 그룹과 전체 뇌졸중 비율 비교")
bmi_missing_df = df[df["bmi"].isna()]
bmi_missing_count = len(bmi_missing_df)
bmi_missing_stroke_count = int(bmi_missing_df["stroke"].sum())
bmi_missing_ratio = (
    (bmi_missing_stroke_count / bmi_missing_count) * 100
    if bmi_missing_count > 0
    else 0
)

total_count = len(df)
total_stroke_count = int(df["stroke"].sum())
total_ratio = (total_stroke_count / total_count) * 100

comparison_data = [
    {
        "구분": "BMI 비어 있는 사람",
        "인원수": bmi_missing_count,
        "뇌졸중 환자수": bmi_missing_stroke_count,
        "뇌졸중 비율(%)": round(bmi_missing_ratio, 2),
    },
    {
        "구분": "전체 데이터",
        "인원수": total_count,
        "뇌졸중 환자수": total_stroke_count,
        "뇌졸중 비율(%)": round(total_ratio, 2),
    },
]
comparison_df = pd.DataFrame(comparison_data)
st.dataframe(comparison_df, use_container_width=True, hide_index=True)

st.divider()

# 5. 흡연 상태의 값마다 사람 수 세어 표로 보여주기
st.subheader("5. 흡연 상태(smoking_status)별 사람 수")
smoking_counts = (
    df["smoking_status"]
    .fillna("Unknown (결측치)")
    .value_counts()
    .reset_index()
)
smoking_counts.columns = ["흡연 상태", "사람 수"]
st.dataframe(smoking_counts, use_container_width=True, hide_index=True)
