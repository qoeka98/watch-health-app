import streamlit as st
import numpy as np
import joblib
import os

# ✅ 모델 파일 확인 후 로드
MODEL_PATH = os.path.join(os.getcwd(), "classifier2_model.pkl")

if not os.path.exists(MODEL_PATH):
    st.error("🚨 모델 파일이 없습니다! `classifier2_model.pkl`을 업로드하거나 경로를 확인하세요.")
    st.stop()

model = joblib.load(MODEL_PATH)

def run_eda():
    st.title("🩺 건강 예측 AI")
    st.markdown("📌 **아래 설문지를 작성하면 AI가 건강 위험도를 예측합니다.**")

    with st.form("health_form"):
        gender = st.radio("🔹 성별", ["여성", "남성"])
        age = st.slider("🔹 나이", 10, 100, 40)
        height = st.number_input("🔹 키 (cm)", min_value=120, max_value=250, value=170)
        weight = st.number_input("🔹 몸무게 (kg)", min_value=30, max_value=200, value=70)
        systolic_bp = st.number_input("💓 수축기 혈압", min_value=50, max_value=200, value=120)
        diastolic_bp = st.number_input("🩸 이완기 혈압", min_value=40, max_value=150, value=80)
        smoke = st.checkbox("🚬 흡연 여부")
        alco = st.checkbox("🍺 음주 여부")
        active = st.checkbox("🏃 운동 여부")
        submit = st.form_submit_button("🔮 예측하기")

    if submit:
        # ✅ 유저 입력값 변환
        input_data = np.array([[
            float(1 if gender == "남성" else 0), 
            float(age), float(height), float(weight),
            float(smoke), float(alco), float(active),
            float(systolic_bp), float(diastolic_bp),
            float(systolic_bp / diastolic_bp),  # BP Ratio
            float(weight / ((height / 100) ** 2)),  # BMI
            float(systolic_bp - diastolic_bp)  # BP Diff
        ]])

        # ✅ 모델 예측 실행
        try:
            predicted_probs = model.predict_proba(input_data)
            if isinstance(predicted_probs, list):
                predicted_probs = np.array(predicted_probs)

            # 🚨 차원이 다를 경우 자동 변환
            if predicted_probs.ndim == 3:
                predicted_probs = predicted_probs.squeeze()

        except Exception as e:
            st.error(f"🚨 모델 예측 중 오류 발생: {e}")
            predicted_probs = np.zeros((4, 2))  # 기본값 0 설정

        diseases = ["고혈압", "비만", "당뇨병", "고지혈증"]

        # ✅ 최종 예측 확률 계산
        disease_probabilities = {}
        for i, disease in enumerate(diseases):
            try:
                prob = predicted_probs[i][1] * 100  
            except IndexError:
                prob = 0  

            if not isinstance(prob, (int, float)) or np.isnan(prob) or np.isinf(prob):
                prob = 0  

            disease_probabilities[disease] = prob

        # ✅ 예측 결과 출력
        st.subheader("🔍 예측된 질병 확률 확인")
        st.json(disease_probabilities)

        for disease, prob in disease_probabilities.items():
            normalized_prob = min(max(prob / 100, 0), 1)
            st.metric(label=f"📊 {disease} 위험", value=f"{prob:.2f}%")
            st.progress(normalized_prob)

if __name__ == "__main__":
    run_eda()
