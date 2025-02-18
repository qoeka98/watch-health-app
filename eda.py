import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go
import xgboost as xgb

# 🔹 모델 로드
model = joblib.load("multioutput_classifier.pkl")

# 🔹 내부 XGBoost 모델 개별 로드
for i in range(len(model.estimators_)):
    booster = xgb.Booster()
    booster.load_model(f"xgb_model_{i}.json")
    model.estimators_[i] = xgb.XGBClassifier()
    model.estimators_[i]._Booster = booster
    model.estimators_[i].n_classes_ = 2  # 이진 분류 문제

# 🔹 고혈압 위험도 계산 함수
def calculate_hypertension_risk(systolic_bp, diastolic_bp, blood_pressure_diff, smoke, alco, active):
    base_risk = 10
    base_risk += max(0, (systolic_bp - 120) * 1.5)
    base_risk += max(0, (diastolic_bp - 80) * 1.2)
    base_risk += max(0, (blood_pressure_diff - 50) * 0.5)
    if smoke == 0: base_risk += 10
    if alco == 0: base_risk += 10
    if active == 0: base_risk -= 10
    return min(max(base_risk, 0), 100)  # 0~100 범위 제한

# 🔹 Streamlit 앱 실행
def run_eda():
    st.title("🩺 건강 예측 AI")
    st.markdown("📌 **아래 설문지를 작성하면 AI가 건강 위험도를 예측합니다.**")

    with st.form("health_form"):
        st.markdown("### 📝 개인정보 입력")
        col1, col2 = st.columns(2)
        with col1:
            gender = st.radio("🔹 성별", ["여성", "남성"])
            age = st.slider("🔹 나이", 10, 100, 40)
        with col2:
            height = st.number_input("🔹 키 (cm)", min_value=120, max_value=250, value=170)
            weight = st.number_input("🔹 몸무게 (kg)", min_value=30, max_value=200, value=70)

        st.markdown("### 💖 건강 정보 입력")
        col3, col4 = st.columns(2)
        with col3:
            systolic_bp = st.number_input("💓 수축기(최고) 혈압 (mmHg)", min_value=50, max_value=200, value=120)
        with col4:
            diastolic_bp = st.number_input("🩸 이완기(최저) 혈압 (mmHg)", min_value=40, max_value=150, value=80)

        st.markdown("### 🏃 생활 습관 입력")
        col5, col6, col7 = st.columns(3)
        with col5:
            smoke = st.checkbox("🚬 흡연 여부")
            smoke = 0 if smoke else 1
        with col6:
            alco = st.checkbox("🍺 음주 여부")
            alco = 0 if alco else 1
        with col7:
            active = st.checkbox("🏃 운동 여부")
            active = 0 if active else 1

        submit = st.form_submit_button("🔮 예측하기")

    if submit:
        bp_ratio = round(systolic_bp / diastolic_bp, 2) if diastolic_bp > 0 else 0
        BMI = round(weight / ((height / 100) ** 2), 2)
        blood_pressure_diff = systolic_bp - diastolic_bp
        hypertension_risk = calculate_hypertension_risk(systolic_bp, diastolic_bp, blood_pressure_diff, smoke, alco, active)

        input_data = np.array([[1 if gender == "남성" else 0, age, height, weight, smoke, alco, active,
                                systolic_bp, diastolic_bp, bp_ratio, BMI, blood_pressure_diff]])

        # 🔹 `predict_proba()` 예측값 가져오기 (모든 예측 값이 2차원인지 체크)
        predicted_probs = model.predict_proba(input_data)

        if isinstance(predicted_probs, list):
            predicted_probs = np.array(predicted_probs)

        # 🔹 `NaN`, `None`, `Inf` 값이 존재하는지 체크
        if np.isnan(predicted_probs).any() or np.isinf(predicted_probs).any():
            st.error("🚨 오류: 모델 예측값에 NaN 또는 Inf 값이 포함되어 있습니다.")
            return

        diseases = ["비만", "당뇨병", "고지혈증"]  # 고혈압은 따로 계산
        disease_probabilities = {}

        for i, disease in enumerate(diseases):
            if predicted_probs.ndim == 3:  # 예측값이 3차원 배열일 경우
                disease_probabilities[disease] = predicted_probs[i][0][1] * 100
            elif predicted_probs.ndim == 2:  # 2차원 배열일 경우
                disease_probabilities[disease] = predicted_probs[i][1] * 100
            else:  # 예측값이 예상과 다르게 나오면 기본값 0 설정
                disease_probabilities[disease] = 0

        disease_probabilities["고혈압"] = hypertension_risk

        # 📌 확률 값 검증 및 NaN 값 처리
        for disease in disease_probabilities:
            if np.isnan(disease_probabilities[disease]):  # NaN 체크
                disease_probabilities[disease] = 0
            disease_probabilities[disease] = min(max(disease_probabilities[disease], 0), 100)  # 0~100 보정

        # 📌 결과 시각화
        st.markdown("### 📢 건강 예측 결과")
        col1, col2 = st.columns(2)
        for i, (disease, value) in enumerate(disease_probabilities.items()):
            with col1 if i % 2 == 0 else col2:
                st.metric(label=f"💡 {disease} 위험", value=f"{value:.2f}%")
                
                # 📌 `st.progress()`가 0~1 범위에서만 동작하도록 보정
                progress_value = min(max(value / 100.0, 0.0), 1.0)
                if np.isnan(progress_value):
                    progress_value = 0.0
                st.progress(progress_value)

        # 📌 건강 진단 메시지
        def show_health_risk(disease, very_high=90, high=75, moderate=50, low=35):
            prob = disease_probabilities[disease]
            if prob > very_high:
                st.error(f"🚨 **{disease} 위험이 매우 높습니다! 즉각적인 관리가 필요합니다.**")
            elif prob > high:
                st.warning(f"⚠️ **{disease} 위험이 높습니다. 생활습관 개선이 필요합니다.**")
            elif prob > moderate:
                st.info(f"ℹ️ **{disease} 위험이 중간 수준입니다. 운동과 식이조절을 고려하세요.**")
            else:
                st.success(f"✅ **{disease} 위험이 낮습니다. 건강한 습관을 유지하세요.**")

        for disease in disease_probabilities:
            show_health_risk(disease)

if __name__ == "__main__":
    run_eda()
