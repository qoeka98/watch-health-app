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

# ✅ 대한민국 평균값 (성별에 따라 다름)
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

def get_user_input():
    """사용자로부터 입력을 받는 함수"""
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
            smoke = st.checkbox("🚬 흡연 여부")
        with col6:
            alco = st.checkbox("🍺 음주 여부")
        with col7:
            active = st.checkbox("🏃 운동 여부")

        submit = st.form_submit_button("🔮 예측하기")

    return submit, gender, age, height, weight, systolic_bp, diastolic_bp, int(smoke), int(alco), int(active)

def predict_disease(input_data):
    """모델을 이용해 질병 위험도를 예측하는 함수"""
    if model:
        try:
            predicted_probs = model.predict_proba(input_data) if hasattr(model, "predict_proba") else model.predict(input_data)

            if isinstance(predicted_probs, list):
                predicted_probs = np.array([float(arr[0, 1]) for arr in predicted_probs])
            elif isinstance(predicted_probs, np.ndarray):
                if predicted_probs.ndim == 3:
                    predicted_probs = predicted_probs[:, 0, 1].flatten()
                elif predicted_probs.ndim == 2:
                    predicted_probs = predicted_probs[:, 1].flatten()
            else:
                st.error(f"⚠️ 예측 결과를 변환할 수 없습니다. 형태: {predicted_probs}")
                return {}

            if len(predicted_probs) < 4:
                st.error(f"⚠️ 모델이 4개의 질병을 예측하지 않습니다. 예측 크기: {len(predicted_probs)}")
                return {}

            diseases = ["고혈압", "비만", "당뇨병", "고지혈증"]
            return {diseases[i]: float(predicted_probs[i] * 100) for i in range(4)}

        except Exception as e:
            st.error(f"⚠️ 예측 오류 발생: {e}")
            return {}

    st.error("⚠️ 모델이 로드되지 않았습니다.")
    return {disease: 0 for disease in ["고혈압", "비만", "당뇨병", "고지혈증"]}

def show_health_risk(disease, prob, very_high=90, high=75, moderate=50, low=35):
    """예측 결과를 바탕으로 건강 상태를 표시하는 함수"""
    if prob > very_high:
        st.error(f"🚨 {disease} 위험이 매우 높습니다! 병원 방문을 추천합니다.")
    elif prob > high:
        st.warning(f"⚠️ {disease} 위험이 높습니다. 생활습관 개선이 필요합니다.")
    elif prob > moderate:
        st.info(f"ℹ️ {disease} 위험이 중간 수준입니다. 건강 관리가 필요합니다.")
    else:
        st.success(f"✅ {disease} 위험이 낮은 편입니다. 건강한 습관을 유지하세요.")

def run_eda():
    st.title("🩺 건강 예측 AI")
    st.markdown("📌 **아래 설문지를 작성하면 AI가 건강 위험도를 예측합니다.**")

    submit, gender, age, height, weight, systolic_bp, diastolic_bp, smoke, alco, active = get_user_input()

    if submit:
        gender_value = 1 if gender == "남성" else 0
        bp_ratio = round(systolic_bp / diastolic_bp, 2) if diastolic_bp > 0 else 0
        BMI = round(weight / ((height / 100) ** 2), 2) if height > 0 else 0
        blood_pressure_diff = systolic_bp - diastolic_bp

        input_data = np.array([[gender_value, age, height, weight, smoke, alco, active, systolic_bp, diastolic_bp, bp_ratio, BMI, blood_pressure_diff]])

        disease_probabilities = predict_disease(input_data)

        st.markdown("### 📢 **건강 예측 결과**")
        for disease, prob in disease_probabilities.items():
            safe_prob = min(1, max(0, prob / 100))
            st.metric(label=f"📊 {disease} 위험", value=f"{prob:.2f}%")
            st.progress(safe_prob)

        st.write("\n### ✅ 건강 진단 및 조치 추천 ✅")
        for disease, prob in disease_probabilities.items():
            show_health_risk(disease, prob)

        # ✅ 평균 비교 차트 생성
        avg_values = avg_values_male if gender == "남성" else avg_values_female
        user_values = {"몸무게 (kg)": weight, "사용자 BMI": BMI, "수축기 혈압": systolic_bp, "이완기 혈압": diastolic_bp}
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(avg_values.keys()), y=list(avg_values.values()), name="대한민국 평균", marker_color="blue", opacity=0.7))
        fig.add_trace(go.Bar(x=list(user_values.keys()), y=list(user_values.values()), name="유저 결과값", marker_color="red", opacity=0.7))
        fig.update_layout(title="📊 평균값과 결과값 비교", xaxis_title="건강 지표", yaxis_title="수치", barmode="group", template="plotly_white")
        st.plotly_chart(fig)

if __name__ == "__main__":
    run_eda()
