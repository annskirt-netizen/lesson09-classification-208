import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# 페이지 설정
st.set_page_config(
    page_title="분류 모델 - 뇌졸중 예측 실습실",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 뇌졸중 예측 분류 모델 실습실")
st.markdown("로지스틱 회귀와 의사결정트리 모델을 학습시키고, 결정 경계와 트리 구조를 시각화합니다.")
st.markdown("---")

# 데이터 로드
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

# 속성 매핑 정의 (영문 열 이름 <-> 우리말 이름)
feature_map = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}
eng_features = list(feature_map.keys())
korean_features = list(feature_map.values())

st.subheader("⚙️ 모델 입력 속성 선택")
# 기본값: bmi를 뺀 넷 (age, avg_glucose_level, hypertension, heart_disease)
default_korean = ["나이", "평균 혈당", "고혈압", "심장병"]

selected_korean = st.multiselect(
    "학습에 사용할 속성을 고르세요 (2개 이상 선택):",
    options=korean_features,
    default=default_korean
)

# 우리말 이름을 영문 열 이름으로 변환
selected_eng = [k for k, v in feature_map.items() if v in selected_korean]

# 2개보다 적게 고르면 안내문 표시 후 멈춤
if len(selected_eng) < 2:
    st.warning("⚠️ 속성을 **2개 이상** 선택해 주세요.")
    st.stop()

# ---------------------------------------------------------
# 데이터 전처리 및 분할 (요구사항 반영)
# ---------------------------------------------------------
# 1. 번호 순(인덱스 순) 정렬
df_sorted = df.sort_index().reset_index(drop=True)

# 2. 10명씩 묶어 각 묶음의 앞 3명을 테스트용으로 고정 (10명 중 index 0, 1, 2가 테스트)
test_indices = []
train_indices = []

for i in range(0, len(df_sorted), 10):
    chunk = range(i, min(i + 10, len(df_sorted)))
    for idx in chunk:
        # 10명 중 모듈러 연산으로 앞 3명(0, 1, 2) 선택
        if (idx % 10) < 3:
            test_indices.append(idx)
        else:
            train_indices.append(idx)

train_df = df_sorted.loc[train_indices].copy()
test_df = df_sorted.loc[test_indices].copy()

# 3. 체질량지수(bmi) 결측치는 훈련용의 중앙값으로 채우기
if "bmi" in selected_eng:
    train_bmi_median = train_df["bmi"].median()
    train_df["bmi"] = train_df["bmi"].fillna(train_bmi_median)
    test_df["bmi"] = test_df["bmi"].fillna(train_bmi_median)

# 모델 학습용 X, y 준비
X_train = train_df[selected_eng]
y_train = train_df["stroke"]
X_test = test_df[selected_eng]
y_test = test_df["stroke"]

# ---------------------------------------------------------
# 모델 학습
# ---------------------------------------------------------
# 1. 로지스틱 회귀 (확률로 답하는 모델)
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)

lr_train_acc = lr_model.score(X_train, y_train)
lr_test_acc = lr_model.score(X_test, y_test)

# 2. 의사결정트리 (질문으로 답하는 모델) - 제약 조건: 질문 3번까지(max_depth=3), 리프 노드 최소 5명(min_samples_leaf=5), 난수 고정
dt_model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
dt_model.fit(X_train, y_train)

dt_train_acc = dt_model.score(X_train, y_train)
dt_test_acc = dt_model.score(X_test, y_test)

# 3. 베이스라인 모델 ("입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델")
majority_class = y_train.mode()[0]
baseline_train_acc = (y_train == majority_class).mean()
baseline_test_acc = (y_test == majority_class).mean()

st.markdown("---")
st.subheader("📈 모델별 정확도 비교")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="로지스틱 회귀 (확률로 답하는 모델)",
        value=f"{lr_test_acc * 100:.2f}%",
        delta=f"훈련 정확도: {lr_train_acc * 100:.2f}%"
    )

with col2:
    st.metric(
        label="의사결정트리 (질문으로 답하는 모델)",
        value=f"{dt_test_acc * 100:.2f}%",
        delta=f"훈련 정확도: {dt_train_acc * 100:.2f}%"
    )

with col3:
    st.metric(
        label="다수 클래스 예측 모델 (베이스라인)",
        value=f"{baseline_test_acc * 100:.2f}%",
        delta=f"훈련 정확도: {baseline_train_acc * 100:.2f}%"
    )

st.markdown("---")

# ---------------------------------------------------------
# 산점도 및 결정 경계 시각화
# ---------------------------------------------------------
st.subheader("🗺️ 모델 결정 경계 및 테스트 데이터 산점도")

if len(selected_eng) < 2:
    st.info("시각화를 위해 속성을 2개 이상 선택해 주세요.")
else:
    # 시각화용 축 선택 (고른 속성 중 처음 두 개 사용)
    x_axis_eng = selected_eng[0]
    y_axis_eng = selected_eng[1]
    x_axis_kor = feature_map[x_axis_eng]
    y_axis_kor = feature_map[y_axis_eng]

    # 두 축이 아닌 나머지 속성들은 테스트 데이터의 중앙값으로 고정
    other_features = [f for f in selected_eng if f not in [x_axis_eng, y_axis_eng]]
    fixed_values = {}
    fixed_desc_list = []
    for f in other_features:
        med_val = test_df[f].median()
        fixed_values[f] = med_val
        fixed_desc_list.append(f"**{feature_map[f]}**: {med_val:.2f}")

    if fixed_desc_list:
        st.markdown(f"ℹ️ 선택된 두 축({x_axis_kor}, {y_axis_kor}) 외의 속성은 테스트 데이터 중앙값으로 고정하여 계산했습니다: " + ", ".join(fixed_desc_list))

    # 테스트 데이터 시각화용 데이터프레임
    vis_test_df = test_df.copy()
    vis_test_df["stroke_label"] = vis_test_df["stroke"].map({0: "아님 (0)", 1: "뇌졸중 (1)"})

    fig = px.scatter(
        vis_test_df,
        x=x_axis_eng,
        y=y_axis_eng,
        color="stroke_label",
        color_discrete_map={"아님 (0)": "#1f77b4", "뇌졸중 (1)": "#d62728"},
        labels={x_axis_eng: x_axis_kor, y_axis_eng: y_axis_kor},
        title=f"테스트 데이터 산점도 및 로지스틱 회귀 0.5 경계선 ({x_axis_kor} vs {y_axis_kor})"
    )

    # 1. 로지스틱 회귀 0.5 경계선 그리기 (Grid 생성)
    x_min, x_max = vis_test_df[x_axis_eng].min(), vis_test_df[x_axis_eng].max()
    y_min, y_max = vis_test_df[y_axis_eng].min(), vis_test_df[y_axis_eng].max()
    
    # 여유 공간 부여
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min_plot, x_max_plot = x_min - x_range * 0.1, x_max + x_range * 0.1
    y_min_plot, y_max_plot = y_min - y_range * 0.1, y_max + y_range * 0.1

    xx, yy = np.meshgrid(np.linspace(x_min_plot, x_max_plot, 200), np.linspace(y_min_plot, y_max_plot, 200))
    
    # 예측용 그리드 데이터 생성 (모든 선택된 속성 포함)
    grid_data = pd.DataFrame(index=np.arange(xx.ravel().shape[0]))
    for f in selected_eng:
        if f == x_axis_eng:
            grid_data[f] = xx.ravel()
        elif f == y_axis_eng:
            grid_data[f] = yy.ravel()
        else:
            grid_data[f] = fixed_values[f]

    # 로지스틱 회귀 예측 확률 계산
    probs = lr_model.predict_proba(grid_data)[:, 1]
    zz = probs.reshape(xx.shape)

    # 등고선 추가 (확률 0.5인 선)
    fig.add_trace(
        go.Contour(
            z=zz,
            x=np.linspace(x_min_plot, x_max_plot, 200),
            y=np.linspace(y_min_plot, y_max_plot, 200),
            contours=dict(start=0.5, end=0.5, size=0, coloring='lines'),
            line=dict(color='black', width=2, dash='dash'),
            name='로지스틱 경계 (0.5)',
            showscale=False
        )
    )

    # 2. 의사결정트리가 나눈 칸을 옅은 색으로 칠하기 (배경 예측)
    dt_preds = dt_model.predict(grid_data)
    zz_dt = dt_preds.reshape(xx.shape)

    fig.add_trace(
        go.Contour(
            z=zz_dt,
            x=np.linspace(x_min_plot, x_max_plot, 200),
            y=np.linspace(y_min_plot, y_max_plot, 200),
            colorscale=[[0, 'rgba(31, 119, 180, 0.1)'], [1, 'rgba(214, 39, 40, 0.1)']],
            showscale=False,
            hoverinfo='skip',
            name='의사결정트리 영역'
        )
    )

    fig.update_layout(
        xaxis_title=x_axis_kor,
        yaxis_title=y_axis_kor,
        legend_title="실제 뇌졸중 여부"
    )

    st.plotly_chart(fig, use_container_width=True)

    # 경계선이 그림 범위 내에 있는지 확인 및 안내
    # (여기서는 시각화 영역 내 데이터 분포 기준 판단 안내)
    st.markdown("📌 **안내**: 점선은 로지스틱 회귀 모델이 확률 0.5로 분류하는 경계선이며, 배경의 옅은 색상은 의사결정트리 모델의 영역 구분을 나타냅니다.")

st.markdown("---")

# ---------------------------------------------------------
# 의사결정트리 가지 그림 (Graphviz DOT)
# ---------------------------------------------------------
st.subheader("🌳 의사결정트리 구조 (가지 그림)")

from sklearn.tree import export_graphviz

# 한글 속성 이름 매핑 리스트 생성
feature_names_kor = [feature_map[f] for f in selected_eng]

# Graphviz DOT 문자열 생성 (한글 레이블 적용)
dot_data = export_graphviz(
    dt_model,
    out_file=None,
    feature_names=feature_names_kor,
    class_names=["아님", "뇌졸중"],
    filled=True,
    rounded=True,
    special_characters=True,
    precision=2
)

# st.graphviz_chart를 이용해 트리 시각화
st.graphviz_chart(dot_data)

# 트리 분석 요약 정보 계산
n_nodes = dt_model.tree_.node_count
# 리프 노드(자식이 없는 노드) 개수 계산
is_leaf = (dt_model.tree_.children_left == -1) & (dt_model.tree_.children_right == -1)
n_leaves = int(is_leaf.sum())

# 리프 노드 중 '아님'으로 예측하는 노드 개수 계산 (다수 클래스가 0인 경우)
leaf_values = dt_model.tree_.value[is_leaf]
leaf_majority = np.argmax(leaf_values, axis=2).squeeze()
n_leaves_not_stroke = int((leaf_majority == 0).sum())

# 실제로 모델이 사용한 속성(인덱스가 -2가 아닌 노드들의 feature)
used_feature_indices = dt_model.tree_.feature
used_feature_indices = used_feature_indices[used_feature_indices != -2]
used_features_kor = set([feature_names_kor[i] for i in used_feature_indices])

st.markdown(f"""
- **답을 내는 마디(리프 노드) 총 개수**: {n_leaves}개
- **그중 '아님'이라고 답하는 마디 개수**: {n_leaves_not_stroke}개
- **고른 속성 가운데 이 나무가 실제로 물은 것**: {", ".join(used_features_kor) if used_features_kor else "없음"}
""")
