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

    with st.form("health_form"):
        st.markdown("### 📝 **개인정보 설문**")
        col1, col2 = st.columns(2)
        with col1:
            gender = st.radio("🔹 성별", ["여성", "남성"])
            age = st.slider("🔹 나이", 10, 100, 40)
        with col2:
            height = st.number_input("🔹 키 (cm)", min_value=120, max_value=250, value=170)
            weight = st.number_input("🔹 몸무게 (kg)", min_value=30, max_value=200, value=70)

        st.markdown("---")
        st.markdown("### 💖 **건강 정보 입력**")
        col3, col4 = st.columns(2)
        with col3:
            systolic_bp = st.number_input("💓 수축기(최고) 혈압 (mmHg)", min_value=50, max_value=200, value=120)
        with col4:
            diastolic_bp = st.number_input("🩸 이완기(최저) 혈압 (mmHg)", min_value=40, max_value=150, value=80)

        st.markdown("---")
        st.markdown("### 🏃 **생활 습관 입력**")
        col5, col6, col7 = st.columns(3)
        with col5:
            smoke = st.checkbox("🚬 흡연을 하지 않습니다", key="smoke_checkbox")
        with col6:
            alco = st.checkbox("🍺 음주를 하지 않습니다", key="alco_checkbox")
        with col7:
            active = st.checkbox("🏃 운동을 일주일에 1시간 이상 꾸준히 합니다", key="active_checkbox")

        submit = st.form_submit_button("🔮 예측하기")

    if submit:
        try:
            # ✅ 입력 데이터 변환
            gender_value = 1 if gender == "남성" else 0
            bp_ratio = round(systolic_bp / diastolic_bp, 2) if diastolic_bp > 0 else 0
            BMI = round(weight / ((height / 100) ** 2), 2) if height > 0 else 0
            blood_pressure_diff = systolic_bp - diastolic_bp

            # ✅ 체크박스 값 변환 (True -> 1, False -> 0)
            smoke_value = float(smoke)  # 흡연 → 1이면 위험 증가
            alco_value = float(alco)  # 음주 → 1이면 위험 증가
            active_value = float(active)  # 운동 → 1이면 위험 감소, 0이면 위험 증가 (변환 없음)

            input_data = np.array([[ 
                gender_value, age, height, weight,
                smoke_value, alco_value, active_value, systolic_bp, diastolic_bp,
                bp_ratio, BMI, blood_pressure_diff
            ]])

            if model:
                if hasattr(model, "predict_proba"):
                    predicted_probs = model.predict_proba(input_data)
                else:
                    predicted_probs = model.predict(input_data)

                # 🔍 예측 결과 변환
                if isinstance(predicted_probs, list):
                    predicted_probs = np.array([float(arr[0, 1]) for arr in predicted_probs])
                elif isinstance(predicted_probs, np.ndarray):
                    if predicted_probs.ndim == 3:
                        predicted_probs = predicted_probs[:, 0, 1].flatten()
                    elif predicted_probs.ndim == 2:
                        predicted_probs = predicted_probs[:, 1].flatten()
                    elif predicted_probs.ndim == 1:
                        pass
                else:
                    st.error(f"⚠️ 예측 결과를 변환할 수 없습니다. 형태: {predicted_probs}")
                    return

                if len(predicted_probs) < 4:
                    st.error(f"⚠️ 모델이 4개의 질병을 예측하지 않습니다. 예측 크기: {len(predicted_probs)}")
                    return

                diseases = ["고혈압", "비만", "당뇨병", "고지혈증"]
                disease_probabilities = {diseases[i]: float(predicted_probs[i] * 100) for i in range(4)}

            else:
                st.error("⚠️ 모델이 로드되지 않아 기본값(0%)을 반환합니다.")
                disease_probabilities = {disease: 0 for disease in ["고혈압", "비만", "당뇨병", "고지혈증"]}

        except Exception as e:
            st.error(f"⚠️ 예측 오류 발생: {e}")
            return

        st.markdown("---")
        st.markdown("### 📢 **건강 예측 결과**")

        for disease, prob in disease_probabilities.items():
            safe_prob = min(1, max(0, prob / 100))  # ✅ 0~1 범위 조정
            st.metric(label=f"📊 {disease} 위험", value=f"{prob:.2f}%")
            st.progress(safe_prob)

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
