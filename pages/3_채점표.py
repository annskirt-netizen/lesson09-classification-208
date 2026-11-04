import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import streamlit as st

# 페이지 설정
st.set_page_config(
    page_title="채점표 및 성능 평가 - 뇌졸중 예측 실습실", page_icon="📋", layout="wide"
)

st.title("📋 뇌졸중 예측 모델 채점표 및 성능 평가")
st.markdown(
    "테스트 데이터를 이용해 두 분류 모델의 혼동행렬(Confusion Matrix)과"
    " 다양한 평가지표(정확도, 재현율, 정밀도, F1 score)를 비교하고,"
    " 오답 사례(놓친 환자 및 헛짚은 사람)를 상세히 살펴보는 공간입니다."
)

# 데이터 로드
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

# 1. 속성 선택 영역 (분류 모델 페이지와 동일하게 기본 4개 속성 고정 또는 전체 활용)
# 사용자가 직접 고를 수도 있도록 동일한 속성 선택 제공
feature_mapping = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}

available_features = list(feature_mapping.keys())
default_features = [
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease",
]

st.subheader("1. 모델 입력 속성 설정")
selected_features_korean = st.multiselect(
    "입력으로 사용할 속성을 선택하세요",
    options=[feature_mapping[f] for f in available_features],
    default=[feature_mapping[f] for f in default_features],
)

reverse_mapping = {v: k for k, v in feature_mapping.items()}
selected_features = [
    reverse_mapping[k] for k in selected_features_korean
]

if len(selected_features) < 2:
  st.warning("⚠️ 속성을 **2개 이상** 선택해 주세요!")
  st.stop()

st.divider()

# 2. 데이터 전처리 및 분할 (분류 모델 페이지와 정확히 동일한 로직)
df_sorted = df.sort_index().reset_index(drop=True)

train_indices = []
test_indices = []

for i in range(0, len(df_sorted), 10):
  chunk = df_sorted.iloc[i : i + 10]
  if len(chunk) < 10:
    split_point = int(len(chunk) * 0.3)
    test_indices.extend(chunk.iloc[:split_point].index.tolist())
    train_indices.extend(chunk.iloc[split_point:].index.tolist())
  else:
    test_indices.extend(chunk.iloc[:3].index.tolist())
    train_indices.extend(chunk.iloc[3:].index.tolist())

train_df = df_sorted.loc[train_indices].copy()
test_df = df_sorted.loc[test_indices].copy()

# 체질량지수(bmi) 결측치 처리: 훈련용의 중앙값으로 채우기
if "bmi" in selected_features:
  train_bmi_median = train_df["bmi"].median()
  train_df["bmi"] = train_df["bmi"].fillna(train_bmi_median)
  test_df["bmi"] = test_df["bmi"].fillna(train_bmi_median)

X_train = train_df[selected_features]
y_train = train_df["stroke"]
X_test = test_df[selected_features]
y_test = test_df["stroke"]

# 테스트용 사람 수와 그중 뇌졸중인 사람 수 안내
total_test_count = len(y_test)
test_stroke_count = int(y_test.sum())
st.info(
    f"📌 **테스트 데이터 정보**: 총 **{total_test_count:,}명** 중 실제 뇌졸중"
    f" 환자는 **{test_stroke_count:,}명**입니다."
)

# 3. 모델 학습
# 3-1. 로지스틱 회귀 모델
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# 3-2. 의사결정트리 모델
dt_model = DecisionTreeClassifier(
    max_depth=3, min_samples_split=5, random_state=42
)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

# 3-3. 베이스라인 모델 (훈련용에서 많은 쪽으로만 답하는 모델)
majority_class = y_train.mode()[0]
baseline_pred = np.full_like(y_test, majority_class)


# 지표 계산 함수 (정확도, 재현율, 정밀도, F1)
def calculate_metrics(y_true, y_pred):
  cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
  tn, fp, fn, tp = cm.ravel()

  total = tn + fp + fn + tp
  accuracy = (tp + tn) / total if total > 0 else 0
  recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # 재현율 (실제 뇌졸중 중 맞춘 비율)
  precision = tp / (tp + fp) if (tp + fp) > 0 else None  # 정밀도 (예측한 뇌졸중 중 맞춘 비율)

  if precision is None or precision == 0 and recall == 0:
    f1 = 0
  elif precision is None:
    f1 = "계산할 수 없음"
  else:
    f1 = (2 * precision * recall) / (precision + recall)

  return tn, fp, fn, tp, accuracy, recall, precision, f1


lr_metrics = calculate_metrics(y_test, lr_pred)
dt_metrics = calculate_metrics(y_test, dt_pred)
base_metrics = calculate_metrics(y_test, baseline_pred)

st.subheader("2. 모델별 혼동행렬 (Confusion Matrix)")

col_cm1, col_cm2 = st.columns(2)


def display_confusion_matrix(tn, fp, fn, tp, model_title):
  cm_df = pd.DataFrame(
      [
          {
              "예측: 정상(0)": f"TN: {tn:,}명",
              "예측: 뇌졸중(1)": f"FP: {fp:,}명",
              "행 합계": f"{tn + fp:,}명",
          },
          {
              "예측: 정상(0)": f"FN: {fn:,}명",
              "예측: 뇌졸중(1)": f"TP: {tp:,}명",
              "행 합계": f"{fn + tp:,}명",
          },
      ],
      index=["실제: 정상(0)", "실제: 뇌졸중(1)"],
  )
  cm_df.loc["열 합계"] = [
      f"{tn + fn:,}명",
      f"{fp + tp:,}명",
      f"{tn + fp + fn + tp:,}명",
  ]
  st.write(f"**{model_title}**")
  st.dataframe(cm_df, use_container_width=True)


with col_cm1:
  display_confusion_matrix(
      lr_metrics[0],
      lr_metrics[1],
      lr_metrics[2],
      lr_metrics[3],
      "로지스틱 회귀(확률로 답하는 모델)",
  )

with col_cm2:
  display_confusion_matrix(
      dt_metrics[0],
      dt_metrics[1],
      dt_metrics[2],
      dt_metrics[3],
      "의사결정트리(질문으로 답하는 모델)",
  )

st.divider()

# 4. 성능 지표 비교 표
st.subheader("3. 주요 평가지표 비교 표")


def format_f1(f1_val):
  if f1_val == "계산할 수 없음":
    return "계산할 수 없음"
  return f"{f1_val:.4f}"


performance_data = [
    {
        "모델 이름": "로지스틱 회귀(확률로 답하는 모델)",
        "정확도(Accuracy)": f"{lr_metrics[4]:.4f}",
        "재현율(Recall)": f"{lr_metrics[5]:.4f}",
        "정밀도(Precision)": (
            f"{lr_metrics[6]:.4f}"
            if lr_metrics[6] is not None
            else "계산할 수 없음"
        ),
        "F1 Score": format_f1(lr_metrics[7]),
    },
    {
        "모델 이름": "의사결정트리(질문으로 답하는 모델)",
        "정확도(Accuracy)": f"{dt_metrics[4]:.4f}",
        "재현율(Recall)": f"{dt_metrics[5]:.4f}",
        "정밀도(Precision)": (
            f"{dt_metrics[6]:.4f}"
            if dt_metrics[6] is not None
            else "계산할 수 없음"
        ),
        "F1 Score": format_f1(dt_metrics[7]),
    },
    {
        "모델 이름": "다수 클래스 기준 모델 (베이스라인)",
        "정확도(Accuracy)": f"{base_metrics[4]:.4f}",
        "재현율(Recall)": f"{base_
