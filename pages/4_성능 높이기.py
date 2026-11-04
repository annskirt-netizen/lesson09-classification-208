import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

# 페이지 설정
st.set_page_config(
    page_title="성능 높이기 - 뇌졸중 예측 실습실",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ 클래스 가중치 조정에 따른 모델 성능 비교")
st.markdown("뇌졸중 데이터의 불균형을 해결하기 위해 가중치를 맞추기(balanced) 전과 후의 결과를 나란히 비교합니다.")
st.markdown("---")

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

@st.cache_data
def load_data(url):
    df = pd.read_csv(url, encoding="utf-8")
    return df

try:
    df = load_data(DATA_URL)
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

feature_map = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}
korean_features = list(feature_map.values())

st.subheader("⚙️ 모델 입력 속성 선택")
default_korean = ["나이", "평균 혈당", "고혈압", "심장병"]

selected_korean = st.multiselect(
    "학습에 사용할 속성을 고르세요 (2개 이상 선택):",
    options=korean_features,
    default=default_korean,
    key="balance_features"
)

selected_eng = [k for k, v in feature_map.items() if v in selected_korean]

if len(selected_eng) < 2:
    st.warning("⚠️ 속성을 **2개 이상** 선택해 주세요.")
    st.stop()

df_sorted = df.sort_index().reset_index(drop=True)

test_indices = []
train_indices = []

for i in range(0, len(df_sorted), 10):
    chunk = range(i, min(i + 10, len(df_sorted)))
    for idx in chunk:
        if (idx % 10) < 3:
            test_indices.append(idx)
        else:
            train_indices.append(idx)

train_df = df_sorted.loc[train_indices].copy()
test_df = df_sorted.loc[test_indices].copy()

if "bmi" in selected_eng:
    train_bmi_median = train_df["bmi"].median()
    train_df["bmi"] = train_df["bmi"].fillna(train_bmi_median)
    test_df["bmi"] = test_df["bmi"].fillna(train_bmi_median)

X_train = train_df[selected_eng]
y_train = train_df["stroke"]
X_test = test_df[selected_eng]
y_test = test_df["stroke"]

total_test_count = len(y_test)

# 모델 학습 (가중치 조정 전 vs 후)
# 1. 로지스틱 회귀
lr_before = LogisticRegression(random_state=42, max_iter=1000)
lr_before.fit(X_train, y_train)
lr_pred_before = lr_before.predict(X_test)

lr_after = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
lr_after.fit(X_train, y_train)
lr_pred_after = lr_after.predict(X_test)

# 2. 의사결정트리
dt_before = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
dt_before.fit(X_train, y_train)
dt_pred_before = dt_before.predict(X_test)

dt_after = DecisionTreeClassifier(class_weight='balanced', max_depth=3, min_samples_leaf=5, random_state=42)
dt_after.fit(X_train, y_train)
dt_pred_after = dt_after.predict(X_test)

def get_cm_counts(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return int(tn), int(fp), int(fn), int(tp)

def calc_metrics_dict(tn, fp, fn, tp):
    total = tn + fp + fn + tp
    acc = (tp + tn) / total if total > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    pred_stroke = tp + fp
    if pred_stroke == 0:
        precision = "계산할 수 없음"
        f1 = "계산할 수 없음"
    else:
        p_val = tp / pred_stroke
        precision = f"{p_val * 100:.2f}%"
        if (p_val + recall) == 0:
            f1 = "계산할 수 없음"
        else:
            f1_val = 2 * p_val * recall / (p_val + recall)
            f1 = f"{f1_val * 100:.2f}%"
    return {
        "정확도": f"{acc * 100:.2f}%",
        "재현율": f"{recall * 100:.2f}%",
        "정밀도": precision,
        "F1": f1,
        "TP": tp, "FN": fn, "FP": fp, "TN": tn
    }

lr_b_m = calc_metrics_dict(*get_cm_counts(y_test, lr_pred_before))
lr_a_m = calc_metrics_dict(*get_cm_counts(y_test, lr_pred_after))
dt_b_m = calc_metrics_dict(*get_cm_counts(y_test, dt_pred_before))
dt_a_m = calc_metrics_dict(*get_cm_counts(y_test, dt_pred_after))

st.markdown("---")
st.subheader("📊 가중치 조정 전·후 성능 비교 표")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 로지스틱 회귀 (확률로 답하는 모델)")
    lr_comp_df = pd.DataFrame([
        {"구분": "조정 전 (기본)", "정확도": lr_b_m["정확도"], "재현율": lr_b_m["재현율"], "정밀도": lr_b_m["정밀도"], "F1": lr_b_m["F1"]},
        {"구분": "조정 후 (balanced)", "정확도": lr_a_m["정확도"], "재현율": lr_a_m["재현율"], "정밀도": lr_a_m["정밀도"], "F1": lr_a_m["F1"]}
    ])
    st.dataframe(lr_comp_df, use_container_width=True, hide_index=True)
    st.markdown(f"👉 **변화 요약**: 찾아낸 환자 수(TP) {lr_b_m['TP']}명 $\rightarrow$ {lr_a_m['TP']}명 | 놓친 환자 수(FN) {lr_b_m['FN']}명 $\rightarrow$ {lr_a_m['FN']}명 | 헛짚은 사람 수(FP) {lr_b_m['FP']}명 $\rightarrow$ {lr_a_m['FP']}명")

with col2:
    st.markdown("#### 의사결정트리 (질문으로 답하는 모델)")
    dt_comp_df = pd.DataFrame([
        {"구분": "조정 전 (기본)", "정확도": dt_b_m["정확도"], "재현율": dt_b_m["재현율"], "정밀도": dt_b_m["정밀도"], "F1": dt_b_m["F1"]},
        {"구분": "조정 후 (balanced)", "정확도": dt_a_m["정확도"], "재현율": dt_a_m["재현율"], "정밀도": dt_a_m["정밀도"], "F1": dt_a_m["F1"]}
    ])
    st.dataframe(dt_comp_df, use_container_width=True, hide_index=True)
    st.markdown(f"👉 **변화 요약**: 찾아낸 환자 수(TP) {dt_b_m['TP']}명 $\rightarrow$ {dt_a_m['TP']}명 | 놓친 환자 수(FN) {dt_b_m['FN']}명 $\rightarrow$ {dt_a_m['FN']}명 | 헛짚은 사람 수(FP) {dt_b_m['FP']}명 $\rightarrow$ {dt_a_m['FP']}명")

st.markdown("---")

# ---------------------------------------------------------
# 산점도 및 결정 경계 시각화 (2차원 또는 3차원)
# ---------------------------------------------------------
st.subheader("🗺️ 가중치 조정 전·후 0.5 경계선 비교 시각화")

vis_model_choice = st.selectbox(
    "시각화할 모델을 선택하세요:",
    options=["로지스틱 회귀 (확률로 답하는 모델)", "의사결정트리 (질문으로 답하는 모델)"]
]

if vis_model_choice.startswith("로지스틱"):
    m_before = lr_before
    m_after = lr_after
else:
    m_before = dt_before
    m_after = dt_after

vis_test_df = test_df.copy()
vis_test_df["stroke_label"] = vis_test_df["stroke"].map({0: "아님 (0)", 1: "뇌졸중 (1)"})

if len(selected_eng) == 2:
    x_axis_eng, y_axis_eng = selected_eng[0], selected_eng[1]
    x_axis_kor, y_axis_kor = feature_map[x_axis_eng], feature_map[y_axis_eng]

    fig = px.scatter(
        vis_test_df,
        x=x_axis_eng,
        y=y_axis_eng,
        color="stroke_label",
        color_discrete_map={"아님 (0)": "#1f77b4", "뇌졸중 (1)": "#d62728"},
        labels={x_axis_eng: x_axis_kor, y_axis_eng: y_axis_kor},
        title=f"테스트 데이터 산점도 및 전·후 0.5 경계선 비교 ({x_axis_kor} vs {y_axis_kor})"
    )

    x_min, x_max = vis_test_df[x_axis_eng].min(), vis_test_df[x_axis_eng].max()
    y_min, y_max = vis_test_df[y_axis_eng].min(), vis_test_df[y_axis_eng].max()
    x_r, y_r = x_max - x_min, y_max - y_min
    xx, yy = np.meshgrid(np.linspace(x_min - x_r*0.1, x
