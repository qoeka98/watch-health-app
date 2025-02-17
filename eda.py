import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go
import os

# ✅ 모델 파일 확인 후 로드
model_path = "classifier2_model.pkl"
if not os.path.exists(model_path):
    st.error("🚨 모델 파일이 없습니다! `classifier2_model.pkl`을 확인하세요.")
    st.stop()

model = joblib.load(model_path)

def run_eda():
    st.title("🩺 건강 예측 AI")
    st.markdown("📌 **아래 설문지를 작성하면 AI가 건강 위험도를 예측합니다.**")
    
    # ✅ 평균값 설정 (BMI 추가)
    avg_values_male = {
        "몸무게 (kg)": 74, "대한민국 평균 BMI": 24.8,
        "수축기 혈압": 120, "이완기 혈압": 78,
        "고혈압 위험": 30, "당뇨병 위험": 15, "고지혈증 위험": 25
    }

    avg_values_female = {
        "몸무게 (kg)": 62, "대한민국 평균 BMI": 24.2,
        "수축기 혈압": 115, "이완기 혈압": 75,
        "고혈압 위험": 28, "당뇨병 위험": 12, "고지혈증 위험": 20
    }

    # ✅ 설문 스타일의 입력 폼
    with st.form("health_form"):
        st.markdown("### 📝 **개인정보 설문**")
        st.info("아래 정보를 입력해주세요. (실제 값이 아닐 경우 예측 정확도가 떨어질 수 있습니다.)")

        col1, col2 = st.columns(2)
        with col1:
            gender = st.radio("🔹 성별", ["여성", "남성"])
            age = st.slider("🔹 나이", 10, 100, 40)

        with col2:
            height = st.number_input("🔹 키 (cm)", min_value=120, max_value=250, value=170)
            weight = st.number_input("🔹 몸무게 (kg)", min_value=30, max_value=200, value=70)

        st.markdown("---")  
        st.markdown("### 💖 **건강 정보 입력!**")

        col3, col4 = st.columns(2)
        with col3:
            systolic_bp = st.number_input("💓 수축기(최고) 혈압 (mmHg)", min_value=50, max_value=200, value=120)
        with col4:
            diastolic_bp = st.number_input("🩸 이완기(최저) 혈압 (mmHg)", min_value=40, max_value=150, value=80)

        st.markdown("---")  
        st.markdown("### 🏃 **생활 습관 입력**")
        st.text('해당되는 부분에 체크해주세요(중복가능)')

        col5, col6, col7 = st.columns(3)
        with col5:
            smoke = st.checkbox("🚬 흡연 여부")
            smoke = 1 if smoke else 0

        with col6:
            alco = st.checkbox("🍺 음주 여부")
            alco = 1 if alco else 0

        with col7:
            active = st.checkbox("🏃 운동 여부")
            active = 1 if active else 0

        submit = st.form_submit_button("🔮 예측하기")

    # ✅ 예측 실행
    if submit:
        # 자동 계산
        bp_ratio = round(systolic_bp / diastolic_bp, 2)
        BMI = round(weight / ((height / 100) ** 2), 2)
        blood_pressure_diff = systolic_bp - diastolic_bp

        input_data = np.array([[ 
            1 if gender == "남성" else 0, age, height, weight,
            smoke, alco, active, systolic_bp, diastolic_bp,
            bp_ratio, BMI, blood_pressure_diff  
        ]])

        # ✅ 예측 실행 (모델 출력 검증)
        predicted_probs = model.predict_proba(input_data)

        # ✅ 모델 예측값 보정
        if isinstance(predicted_probs, list):
            predicted_probs = np.array(predicted_probs)  

        if predicted_probs.ndim == 3:
            predicted_probs = predicted_probs.squeeze()  

        # ✅ 질병 리스트
        diseases = ["고혈압", "비만", "당뇨병", "고지혈증"]

        # `predicted_probs` 길이 확인 후 보정
        if len(predicted_probs) < len(diseases):
            print("🚨 모델의 출력 개수가 예상보다 적습니다. 기본값 0으로 채웁니다.")
            predicted_probs = np.pad(predicted_probs, ((0, len(diseases) - len(predicted_probs)), (0, 0)), mode='constant', constant_values=0)

        # ✅ 최종 예측 확률 계산
        disease_probabilities = {}
        for i, disease in enumerate(diseases):
            try:
                prob = predicted_probs[i][1] * 100
            except IndexError:
                print(f"🚨 {disease} 예측값이 없습니다. 기본값 0으로 설정")
                prob = 0

            # `NaN`, `None`, `inf` 방지
            if not isinstance(prob, (int, float)) or np.isnan(prob) or np.isinf(prob):
                prob = 0  

            disease_probabilities[disease] = prob

        # ✅ 예측된 질병 확률 출력 (디버깅용)
        st.subheader("🔍 예측된 질병 확률 확인")
        st.json(disease_probabilities)

        st.markdown("---")  
        st.markdown("### 📢 **건강 예측 결과**")

        col1, col2 = st.columns(2)
        for i, disease in enumerate(diseases):
            prob = disease_probabilities.get(disease, 0)  
            normalized_prob = min(max(prob / 100, 0), 1)  

            if i % 2 == 0:
                with col1:
                    st.metric(label=f"📊 {disease} 위험", value=f"{prob:.2f}%")
                    st.progress(normalized_prob)
            else:
                with col2:
                    st.metric(label=f"📊 {disease} 위험", value=f"{prob:.2f}%")
                    st.progress(normalized_prob)

if __name__ == "__main__":
    run_eda()
