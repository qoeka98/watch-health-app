import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go

# ✅ 모델 불러오기
try:
    model = joblib.load("classifier2_model.pkl")
except Exception as e:
    st.error(f"⚠️ 모델 로딩 실패: {e}")
    model = None

def run_eda():
    st.title("🩺 건강 예측 AI")
    st.markdown("📌 **아래 설문지를 작성하면 AI가 건강 위험도를 예측합니다.**")

    # ✅ 설문 입력 폼
    with st.form("health_form"):
        st.markdown("### 📝 **개인정보 설문**")
        st.info("아래 정보를 입력해주세요. (실제 값이 아닐 경우 예측 정확도가 떨어질 수 있습니다.)")

        col1, col2 = st.columns(2)
        with col1:
            gender = st.radio("🔹 성별", ["여성", "남성"], key="gender")
            age = st.slider("🔹 나이", 10, 100, 40, key="age")

        with col2:
            height = st.number_input("🔹 키 (cm)", min_value=120, max_value=250, value=170, key="height")
            weight = st.number_input("🔹 몸무게 (kg)", min_value=30, max_value=200, value=70, key="weight")

        st.markdown("---")
        st.markdown("### 💖 **건강 정보 입력**")

        col3, col4 = st.columns(2)
        with col3:
            systolic_bp = st.number_input("💓 수축기(최고) 혈압 (mmHg)", min_value=50, max_value=200, value=120, key="systolic_bp")
        with col4:
            diastolic_bp = st.number_input("🩸 이완기(최저) 혈압 (mmHg)", min_value=40, max_value=150, value=80, key="diastolic_bp")

        st.markdown("---")
        st.markdown("### 🏃 **생활 습관 입력**")

        col5, col6, col7 = st.columns(3)
        with col5:
            smoke = st.checkbox("🚬 흡연 여부", key="smoke")
        with col6:
            alco = st.checkbox("🍺 음주 여부", key="alco")
        with col7:
            active = st.checkbox("🏃 운동 여부", key="active")

        submit = st.form_submit_button("🔮 예측하기")

    if submit:
        try:
            # ✅ 사용자 입력 값을 `st.session_state`에서 가져옴
            gender = st.session_state.gender
            age = st.session_state.age
            height = st.session_state.height
            weight = st.session_state.weight
            systolic_bp = st.session_state.systolic_bp
            diastolic_bp = st.session_state.diastolic_bp
            smoke = int(st.session_state.smoke)
            alco = int(st.session_state.alco)
            active = int(st.session_state.active)

            # ✅ 계산된 변수
            bp_ratio = round(systolic_bp / diastolic_bp, 2) if diastolic_bp > 0 else 0
            BMI = round(weight / ((height / 100) ** 2), 2) if height > 0 else 0
            blood_pressure_diff = systolic_bp - diastolic_bp

            # ✅ 모델 입력 데이터 변환
            input_data = np.array([[ 
                1 if gender == "남성" else 0, age, height, weight,
                smoke, alco, active, systolic_bp, diastolic_bp,
                bp_ratio, BMI, blood_pressure_diff
            ]])

            # ✅ 입력값 확인
            st.write("📌 **입력된 데이터:**", input_data)

            # ✅ 예측 실행
            if model:
                if hasattr(model, "predict_proba"):
                    predicted_probs = model.predict_proba(input_data)
                else:
                    predicted_probs = model.predict(input_data)

                # 🔍 **예측 결과 형태 확인**
                if isinstance(predicted_probs, list):  
                    predicted_probs = np.array(predicted_probs)  # 리스트 → NumPy 배열 변환
                
                if predicted_probs.ndim == 1:  
                    predicted_probs = predicted_probs.reshape(1, -1)  # (N,) → (1, N) 변환
                
                # 🔍 모델이 4개의 질병을 예측하는지 확인
                if predicted_probs.shape[1] < 4:
                    st.error(f"⚠️ 모델이 4개의 질병을 예측하지 않습니다. 예측 크기: {predicted_probs.shape}")
                    return

                diseases = ["고혈압", "비만", "당뇨병", "고지혈증"]
                disease_probabilities = {diseases[i]: predicted_probs[0][i] * 100 for i in range(4)}

            else:
                st.error("⚠️ 모델이 로드되지 않아 기본값(0%)을 반환합니다.")
                disease_probabilities = {disease: 0 for disease in ["고혈압", "비만", "당뇨병", "고지혈증"]}

        except Exception as e:
            st.error(f"⚠️ 예측 오류 발생: {e}")
            return

        st.markdown("---")
        st.markdown("### 📢 **건강 예측 결과**")

        for disease, prob in disease_probabilities.items():
            st.metric(label=f"📊 {disease} 위험", value=f"{prob:.2f}%")
            st.progress(prob / 100)

        st.write("\n### ✅ 건강 진단 및 조치 추천 ✅")

        def show_health_risk(disease, very_high=90, high=75, moderate=50, low=35):
            prob = disease_probabilities[disease]
            if prob > very_high:
                st.error(f"🚨 {disease} 위험이 매우 높습니다! 병원 방문을 추천합니다.")
            elif prob > high:
                st.warning(f"⚠️ {disease} 위험이 높습니다. 생활습관 개선이 필요합니다.")
            elif prob > moderate:
                st.info(f"ℹ️ {disease} 위험이 중간 수준입니다. 건강 관리가 필요합니다.")
            else:
                st.success(f"✅ {disease} 위험이 낮은 편입니다. 건강한 습관을 유지하세요.")

        for disease in disease_probabilities:
            show_health_risk(disease)

if __name__ == "__main__":
    run_eda()
