import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go

# 고혈압 위험도 계산 함수
def calculate_hypertension_risk(systolic_bp, diastolic_bp, smoke, alco, active):
    """
    고혈압 위험도를 혈압 수치와 라이프스타일에 기반하여 직접 계산하는 함수.
    기준: 최고혈압 120, 최저혈압 80일 때 기본 위험은 10%
    """
    if systolic_bp >= 140 or diastolic_bp >= 90:
        base_risk = 80  # 고혈압 기준 초과 시 높은 위험
    elif systolic_bp >= 130 or diastolic_bp >= 85:
        base_risk = 60  # 경계성 고혈압
    elif systolic_bp >= 120 or diastolic_bp >= 80:
        base_risk = 40  # 약간 높은 위험
    else:
        base_risk = 20  # 정상 범위
    
    # 라이프스타일 보정: (여기서는 0이면 해당 활동이 있었음을 의미)
    if smoke == 0:   # 흡연한 경우
        base_risk += 10
    if alco == 0:    # 음주한 경우
        base_risk += 10
    if active == 0:  # 운동한 경우
        base_risk -= 10

    return min(max(base_risk, 0), 100)

# 모델 불러오기
model = joblib.load("classifier2_model.pkl")

def run_eda():
    st.title("🩺 건강 예측 AI")
    st.markdown("📌 아래 정보를 입력하면 혈압, 비만, 당뇨병, 고지혈증의 위험도를 예측합니다.")

    # 평균값 (비교용, 남/여 기준)
    avg_values_male = {
        "나이": 45,
        "키 (cm)": 172,
        "몸무게 (kg)": 74,
        "최고혈압": 120,
        "최저혈압": 78,
        "고혈압 위험": 30,
        "당뇨병 위험": 15,
        "고지혈증 위험": 25,
        "대한민국 평균 BMI": 24.8
    }
    avg_values_female = {
        "나이": 45,
        "키 (cm)": 160,
        "몸무게 (kg)": 62,
        "최고혈압": 115,
        "최저혈압": 75,
        "고혈압 위험": 28,
        "당뇨병 위험": 12,
        "고지혈증 위험": 20,
        "대한민국 평균 BMI": 24.2
    }

    # 사용자 입력 폼 (원시 입력: 총 9개)
    with st.form("health_form"):
        st.markdown("### 📝 개인정보 입력")
        st.info("아래 정보를 입력해주세요.")
        col1, col2 = st.columns(2)
        with col1:
            gender = st.radio("🔹 성별", ["여성", "남성"])
            age = st.slider("🔹 나이", 10, 100, 40)
        with col2:
            height = st.number_input("🔹 키 (cm)", min_value=120, max_value=250, value=170)
            weight = st.number_input("🔹 몸무게 (kg)", min_value=30, max_value=200, value=70)
        
        st.markdown("---")
        st.markdown("### 💖 건강 관련 정보")
        col3, col4 = st.columns(2)
        with col3:
            systolic_bp = st.number_input("💓 최고 혈압 (mmHg)", min_value=50, max_value=200, value=120)
        with col4:
            diastolic_bp = st.number_input("🩸 최저 혈압 (mmHg)", min_value=40, max_value=150, value=80)
        
        st.markdown("---")
        st.markdown("### 🏃 생활 습관 입력")
        st.write("해당 부분에 체크해주세요 (복수 선택 가능)")
        col5, col6, col7 = st.columns(3)
        # 모델 학습 시 인코딩: 0 = 있음, 1 = 없음
        with col5:
            smoke_input = st.checkbox("🚬 흡연 여부")
            smoke = 0 if smoke_input else 1
        with col6:
            alco_input = st.checkbox("🍺 음주 여부")
            alco = 0 if alco_input else 1
        with col7:
            active_input = st.checkbox("🏃 운동 여부")
            active = 0 if active_input else 1
        
        st.write("-----")
        submit = st.form_submit_button("🔮 예측하기")
    
    if submit:
        # [1] 입력 전처리: 원시 입력 9개 → 7개 피처 계산
        gender_val = 1 if gender == "남성" else 0
        bp_diff = systolic_bp - diastolic_bp
        BMI_val = weight / ((height / 100) ** 2)
        # 생활습관 점수: (음주 여부 + 흡연 여부) - 운동 여부 (체크된 원래 값 사용)
        lifestyle_score = (1 if alco_input else 0) + (1 if smoke_input else 0) - (1 if active_input else 0)
        
        features = np.array([[gender_val, age, height, weight, bp_diff, BMI_val, lifestyle_score]], dtype=np.float32)
        feature_names = ["gender", "age", "height", "weight", "bp_diff", "BMI", "lifestyle_score"]
        df_features = pd.DataFrame(features, columns=feature_names)
        
        # [2] 모델 예측 (고혈압은 별도 계산하므로, 나머지 질환만 예측)
        preds = []
        if hasattr(model, "estimators_"):
            for i, estimator in enumerate(model.estimators_):
                # 첫 번째 추정기가 고혈압이라면 건너뜁니다.
                if i == 0:
                    continue
                preds.append(estimator.predict_proba(df_features))
        else:
            preds = model.predict_proba(df_features)
        
        conditions = ["비만", "당뇨병", "고지혈증"]
        results = {}
        if isinstance(preds, list):
            for cond, pred in zip(conditions, preds):
                results[cond] = pred[0, 1] * 100
        else:
            for i, cond in enumerate(conditions):
                results[cond] = preds[i, 1] * 100
        
        # [3] 비만 위험도 재계산 (BMI 기반)
        if BMI_val <= 16:
            obesity_risk = 5
        elif BMI_val <= 25:
            obesity_risk = ((BMI_val - 16) / (25 - 16)) * (50 - 5) + 5
        elif BMI_val <= 40:
            obesity_risk = ((BMI_val - 25) / (40 - 25)) * (100 - 50) + 50
        else:
            obesity_risk = 100
        results["비만"] = obesity_risk
        
        # [4] 당뇨병 및 고지혈증 위험 반전 (모델 예측값이 높으면 위험이 낮다고 가정)
        for cond in ["당뇨병", "고지혈증"]:
            results[cond] = 100 - results[cond]
        
        # [5] 고혈압 위험도 계산 (혈압 입력 기반 휴리스틱)
        hypertension_risk = calculate_hypertension_risk(systolic_bp, diastolic_bp, smoke, alco, active)
        results["고혈압"] = hypertension_risk
        
        # [6] 라이프스타일 보정 적용 (추가 보정, 필요시 조정)
        # 여기서는 고혈압은 이미 계산했으므로, 당뇨병과 고지혈증에 적용
        for cond in ["당뇨병", "고지혈증"]:
            adj = results[cond]
            if smoke_input:  # 흡연 시 위험 증가
                adj += 5
            if active_input:  # 운동 시 위험 감소
                adj -= 10
            results[cond] = min(max(adj, 0), 100)
        
        # [7] 나이 보정 적용 (기준 나이 50세, 70세 이상은 70세로 고정)
        effective_age = age if age <= 70 else 70
        for cond in results:
            if cond == "고혈압":
                results[cond] = min(max(results[cond] + 0.5 * (effective_age - 50), 0), 100)
            else:
                results[cond] = min(max(results[cond] + (effective_age - 50), 0), 100)
        
        # [8] 최종 결과 출력 및 시각화
        st.write("### 예측 결과")
        for cond, prob in results.items():
            st.metric(label=f"{cond} 위험도", value=f"{prob:.2f}%")
        
        fig = go.Figure(data=[go.Bar(name="예측 결과", x=list(results.keys()), y=list(results.values()))])
        fig.update_layout(title="예측 위험도", xaxis_title="질환", yaxis_title="위험도 (%)")
        st.plotly_chart(fig)

if __name__ == "__main__":
    run_eda()
