import joblib
import numpy as np
import streamlit as st
from scipy.special import expit  # 시그모이드 함수

# 모델 로드
model = joblib.load("multioutput_classifier.pkl")

def sigmoid_scaling(x):
    return expit((x - 0.5) * 3) * 100  # 백분율 스케일 조정

def run_eda():
    st.title("🩺 건강 예측 AI")
    st.markdown("📌 **아래 정보를 입력하면 AI가 건강 위험도를 예측합니다.**")

    # 🔹 유저 입력 받기
    with st.form("health_form"):
        st.subheader("📌 기본 정보 입력")
        col1, col2 = st.columns(2)
        with col1:
            gender = st.radio("🔹 성별", ["여성", "남성"])
            age = st.slider("🔹 나이", 10, 100, 40)
        with col2:
            height = st.number_input("🔹 키 (cm)", min_value=120, max_value=250, value=170)
            weight = st.number_input("🔹 몸무게 (kg)", min_value=30, max_value=200, value=70)

        st.subheader("💖 건강 수치 입력")
        col3, col4 = st.columns(2)
        with col3:
            systolic_bp = st.number_input("💓 수축기 혈압 (mmHg)", min_value=50, max_value=200, value=120)
        with col4:
            diastolic_bp = st.number_input("🩸 이완기 혈압 (mmHg)", min_value=40, max_value=150, value=80)

        st.subheader("🏃 생활 습관 입력")
        col5, col6, col7 = st.columns(3)
        with col5:
            smoke = st.checkbox("🚬 흡연 여부")
        with col6:
            alco = st.checkbox("🍺 음주 여부")
        with col7:
            active = st.checkbox("🏃 운동 여부")

        submit = st.form_submit_button("🔮 예측하기")

    # 🔹 예측 실행
    if submit:
        # 🔹 입력 전처리
        gender_val = 1 if gender == "남성" else 0
        smoke_val = 0 if smoke else 1
        alco_val = 0 if alco else 1
        active_val = 1 if active else 0
        bp_ratio = round(systolic_bp / diastolic_bp, 2) if diastolic_bp > 0 else 0
        BMI = round(weight / ((height / 100) ** 2), 2)
        blood_pressure_diff = systolic_bp - diastolic_bp

        input_data = np.array([[ 
            gender_val, age, height, weight,
            smoke_val, alco_val, active_val, 
            systolic_bp, diastolic_bp, bp_ratio, BMI, blood_pressure_diff
        ]])

        # 🔹 예측 확률 계산
        predicted_probs = np.array(model.predict_proba(input_data))
        print("📌 예측 확률 결과 형태:", predicted_probs.shape)
        print("📌 예측 확률 값:", predicted_probs)

        # 🔹 예측 확률 변환
        diseases = ["고혈압", "비만", "당뇨병", "고지혈증"]
        disease_probabilities = {}

        for i, disease in enumerate(diseases):
            if predicted_probs.ndim == 3:
                disease_probabilities[disease] = predicted_probs[i][0][1] * 100
            elif predicted_probs.ndim == 2:
                disease_probabilities[disease] = predicted_probs[i][1] * 100
            else:
                disease_probabilities[disease] = 0  # 오류 방지

        # 🔹 NaN 값 방지
        for disease in disease_probabilities:
            disease_probabilities[disease] = np.nan_to_num(disease_probabilities[disease], nan=0.0)

        # ✅ 비만 확률 보정 (BMI 기반)
        predicted_obesity = disease_probabilities["비만"]
        if BMI <= 16:
            obesity_risk = 5
        elif BMI <= 23:
            obesity_risk = ((BMI - 16) / (25 - 16)) * (50 - 5) + 5
        elif BMI <= 40:
            obesity_risk = ((BMI - 25) / (40 - 25)) * (100 - 50) + 50
        else:
            obesity_risk = 100

        # 기존 예측과 BMI 기반 예측을 평균 내어 조정
        disease_probabilities["비만"] = (predicted_obesity + obesity_risk) / 2

        # ✅ 비만 확률이 최소 10% 이하로 낮아지지 않도록 조정
        disease_probabilities["비만"] = max(disease_probabilities["비만"], 10)

        # ✅ 다른 질병 확률이 올라갈 때 비만이 비정상적으로 감소하지 않도록 조정
        if disease_probabilities["고혈압"] > 50 or disease_probabilities["당뇨병"] > 50:
            disease_probabilities["비만"] += 5  # 보정 값 추가

        # 🔹 UI 출력
        st.markdown("---")
        st.markdown("### 📢 **건강 예측 결과 (백분율 %)**")

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="💓 고혈압 위험", value=f"{disease_probabilities['고혈압']:.2f}%")
            st.progress(min(max(disease_probabilities["고혈압"] / 100, 0.0), 1.0))
            st.metric(label="⚖️ 비만 위험", value=f"{disease_probabilities['비만']:.2f}%")
            st.progress(min(max(disease_probabilities["비만"] / 100, 0.0), 1.0))
        with col2:
            st.metric(label="🍬 당뇨병 위험", value=f"{disease_probabilities['당뇨병']:.2f}%")
            st.progress(min(max(disease_probabilities["당뇨병"] / 100, 0.0), 1.0))
            st.metric(label="🩸 고지혈증 위험", value=f"{disease_probabilities['고지혈증']:.2f}%")
            st.progress(min(max(disease_probabilities["고지혈증"] / 100, 0.0), 1.0))

if __name__ == "__main__":
    run_eda()
