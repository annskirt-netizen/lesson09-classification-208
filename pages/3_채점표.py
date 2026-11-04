import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

# 페이지 설정
st.set_page_config(
    page_title="채점표 - 뇌졸중 예측 실습실",
    page_icon="📋",
    layout="wide"
)

st.title("📋 모델 채점표 및 평가")
st.markdown("혼동행렬(Confusion Matrix)과 주요 평가 지표를 비교하고, 오답 사례를 분석합니다.")
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
    key="score_features"
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
actual_stroke_test_count = int(y_test.sum())
st.markdown(f"📌 **테스트용 사람 수**: **{total_test_count:,}명** (그중 실제 뇌졸중인 사람 수: **{actual_stroke_test_count:,}명**)")
st.markdown("---")

lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

dt_model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

majority_class = y_train.mode()[0]
baseline_pred = np.full_like(y_test, majority_class)

def get_cm_counts(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return int(tn), int(fp), int(fn), int(tp)

lr_tn, lr_fp, lr_fn, lr_tp = get_cm_counts(y_test, lr_pred)
dt_tn, dt_fp, dt_fn, dt_tp = get_cm_counts(y_test, dt_pred)
base_tn, base_fp, base_fn, base_tp = get_cm_counts(y_test, baseline_pred)

st.subheader("📊 모델별 혼동행렬 (Confusion Matrix)")
st.markdown("행은 **실제 값**, 열은 **예측 값**을 나타냅니다.")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**로지스틱 회귀(확률로 답하는 모델)**")
    lr_cm_df = pd.DataFrame(
        [
            [f"TN: {lr_tn:,}", f"FP: {lr_fp:,}", f"합계: {lr_tn + lr_fp:,}"],
            [f"FN: {lr_fn:,}", f"TP: {lr_tp:,}", f"합계: {lr_fn + lr_tp:,}"],
            [f"합계: {lr_tn + lr_fn:,}", f"합계: {lr_fp + lr_tp:,}", f"전체: {total_test_count:,}"]
        ],
        index=["실제: 아님(0)", "실제: 뇌졸중(1)", "합계"],
        columns=["예측: 아님(0)", "예측: 뇌졸중(1)", "합계"]
    )
    st.dataframe(lr_cm_df, use_container_width=True)

with col2:
    st.markdown("**의사결정트리(질문으로 답하는 모델)**")
    dt_cm_df = pd.DataFrame(
        [
            [f"TN: {dt_tn:,}", f"FP: {dt_fp:,}", f"합계: {dt_tn + dt_fp:,}"],
            [f"FN: {dt_fn:,}", f"TP: {dt_tp:,}", f"합계: {dt_fn + dt_tp:,}"],
            [f"합계: {dt_tn + dt_fn:,}", f"합계: {dt_fp + dt_tp:,}", f"전체: {total_test_count:,}"]
        ],
        index=["실제: 아님(0)", "실제: 뇌졸중(1)", "합계"],
        columns=["예측: 아님(0)", "예측: 뇌졸중(1)", "합계"]
    )
    st.dataframe(dt_cm_df, use_container_width=True)

st.markdown("---")

st.subheader("🏆 모델 성능 지표 비교")

def calc_metrics(tn, fp, fn, tp):
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    predicted_stroke = tp + fp
    if predicted_stroke == 0:
        precision = "계산할 수 없음"
        f1 = "계산할 수 없음"
    else:
        precision_val = tp / predicted_stroke
        if (precision_val + recall) == 0:
            f1 = "계산할 수 없음"
            f1_val = 0.0
        else:
            f1_val = 2 * precision_val * recall / (precision_val + recall)
        precision = f"{precision_val * 100:.2f}%"
        f1 = f"{f1_val * 100:.2f}%"

    return {
        "정확도 (%)": f"{accuracy * 100:.2f}%",
        "재현율 (%)": f"{recall * 100:.2f}%",
        "정밀도": precision,
        "F1 스코어": f1
    }

metrics_table = pd.DataFrame([
    {"모델 구분": "로지스틱 회귀 (확률로 답하는 모델)", **calc_metrics(lr_tn, lr_fp, lr_fn, lr_tp)},
    {"모델 구분": "의사결정트리 (질문으로 답하는 모델)", **calc_metrics(dt_tn, dt_fp, dt_fn, dt_tp)},
    {"모델 구분": "다수 클래스 예측 모델 (베이스라인)", **calc_metrics(base_tn, base_fp, base_fn, base_tp)}
])

st.dataframe(metrics_table, use_container_width=True)

st.markdown("---")

st.subheader("🔍 오답 환자 목록 분석")

chosen_model_name = st.selectbox(
    "오답 목록을 확인할 모델을 선택하세요:",
    options=["로지스틱 회귀 (확률로 답하는 모델)", "의사결정트리 (질문으로 답하는 모델)"]
]

if chosen_model_name.startswith("로지스틱"):
    chosen_pred = lr_pred
    chosen_fn = lr_fn
    chosen_fp = lr_fp
else:
    chosen_pred = dt_pred
    chosen_fn = dt_fn
    chosen_fp = dt_fp

eval_df = test_df.copy()
eval_df["pred"] = chosen_pred

fn_df = eval_df[(eval_df["stroke"] == 1) & (eval_df["pred"] == 0)].copy()
fp_df = eval_df[(eval_df["stroke"] == 0) & (eval_df["pred"] == 1)].copy()

fn_df = fn_df.sort_values(by="age", ascending=False).reset_index(drop=True)
fp_df = fp_df.sort_values(by="age", ascending=False).reset_index(drop=True)

fn_total_count = len(fn_df)
fp_total_count = len(fp_df)

fn_display_df = fn_df.head(20).copy()
fp_display_df = fp_df.head(20).copy()

col_fn, col_fp = st.columns(2)

with col_fn:
    st.markdown(f"#### 놓친 환자 목록 (실제 뇌졸중인데 아니라고 예측) - 전체 {fn_total_count}명")
    if fn_total_count > 20:
        st.caption("⚠️ 인원이 20명을 초과하여 상위 20명만 표시합니다.")
    if not fn_display_df.empty:
        display_fn_table = pd.DataFrame({
            "번호": range(1, len(fn_display_df) + 1),
            "나이": fn_display_df["age"],
            "평균 혈당": fn_display_df["avg_glucose_level"],
            "고혈압 여부": fn_display_df["hypertension"].map({0: "없음", 1: "있음"}),
            "심장병 여부": fn_display_df["heart_disease"].map({0: "없음", 1: "있음"})
        })
        st.dataframe(display_fn_table, use_container_width=True, hide_index=True)
    else:
        st.info("놓친 환자가 없습니다.")

with col_fp:
    st.markdown(f"#### 헛짚은 사람 목록 (실제 아닌데 뇌졸중이라고 예측) - 전체 {fp_total_count}명")
    if fp_total_count > 20:
        st.caption("⚠️ 인원이 20명을 초과하여 상위 20명만 표시합니다.")
    if not fp_display_df.empty:
        display_fp_table = pd.DataFrame({
            "번호": range(1, len(fp_display_df) + 1),
            "나이": fp_display_df["age"],
            "평균 혈당": fp_display_df["avg_glucose_level"],
            "고혈압 여부": fp_display_df["hypertension"].map({0: "없음", 1: "있음"}),
            "심장병 여부": fp_display_df["heart_disease"].map({0: "없음", 1: "있음"})
        })
        st.dataframe(display_fp_table, use_container_width=True, hide_index=True)
    else:
        st.info("헛짚은 사람이 없습니다.")

st.markdown(f"📌 **확인**: 위 목록의 사람 수({fn_total_count}명, {fp_total_count}명)는 혼동행렬의 **FN(거짓 음성)** 및 **FP(거짓 양성)** 칸의 수와 정확히 일치합니다.")
