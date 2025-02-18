import joblib
import numpy as np
import streamlit as st
import pandas as pd

# ✅ BMI 계산 함수
def calculate_bmi(weight, height):
    if height > 0:
        return round(weight / ((height / 100) ** 2), 2)
    return 0

# ✅ 혈압 차 계산 함수
def calculate_bp_difference(systolic_bp, diastolic_bp):
    return systolic_bp - diastolic_bp

# ✅ Min-Max Scaling 함수 (비만 확률 조정용)
def min_max_scale(value, min_val=18.5, max_val=40):
    return max(0, min(100, ((value - min_val) / (max_val - min_val)) * 100))

def run_eda():
    st.title("🩺 건강 예측 AI")
    st.markdown("📌 **건강 정보를 입력하면 AI가 질병 발생 확률을 예측합니다.**")

    # ✅ 사용자 입력 폼
    with st.form("user_input_form"):
        gender = st.radio("🔹 성별", ["여성", "남성"])
        age = st.slider("🔹 나이", 10, 100, 40)
        height = st.number_input("🔹 키 (cm)", min_value=120, max_value=250, value=170)
        weight = st.number_input("🔹 몸무게 (kg)", min_value=30, max_value=200, value=70)
        systolic_bp = st.number_input("💓 수축기 혈압 (mmHg)", min_value=50, max_value=200, value=120)
        diastolic_bp = st.number_input("🩸 이완기 혈압 (mmHg)", min_value=40, max_value=150, value=80)
        
        smoke = 1 if st.checkbox("🚬 흡연 여부") else 0
        alco = 1 if st.checkbox("🍺 음주 여부") else 0
        active = 1 if st.checkbox("🏃 운동 여부") else 0

        # ✅ 폼 제출 버튼
        submit = st.form_submit_button("🔮 예측하기")

    if submit:
        # ✅ BMI 및 혈압차 자동 계산
        BMI = calculate_bmi(weight, height)
        blood_pressure_diff = calculate_bp_difference(systolic_bp, diastolic_bp)
        bp_ratio = round(systolic_bp / diastolic_bp, 2) if diastolic_bp > 0 else 0

        # ✅ BMI 확률 보정 (Min-Max Scaling 적용)
        scaled_BMI = min_max_scale(BMI)

        # ✅ 계산된 값 확인 (디버깅용)
        st.write(f"📌 **계산된 BMI:** {BMI} (보정값: {scaled_BMI}%)")
        st.write(f"📌 **계산된 혈압 차:** {blood_pressure_diff}")
        st.write(f"📌 **계산된 혈압 비율:** {bp_ratio}")

        # ✅ 유저 입력을 기반으로 데이터 생성
        input_data = np.array([[1 if gender == "남성" else 0, age, height, weight, 
                                smoke, alco, active, systolic_bp, diastolic_bp, 
                                bp_ratio, BMI, blood_pressure_diff]])
        
        st.write("📌 모델 입력 데이터:", input_data)

        # ✅ 모델 로드
        model = joblib.load("multioutput_classifier.pkl")

        # ✅ 예측 수행
        predicted_probs = np.array(model.predict_proba(input_data))

        # 🔹 3D 배열일 경우 2D로 변환
        if predicted_probs.ndim == 3:
            predicted_probs = predicted_probs.squeeze(axis=1)  # (4,2) 형태로 변경

        # 📌 예측 확률 결과를 데이터프레임으로 변환
        diseases = ["고혈압", "비만", "당뇨병", "고지혈증"]
        prob_df = {diseases[i]: predicted_probs[i][1] * 100 for i in range(len(diseases))}  # 양성 확률 (1) 만 출력

        # ✅ 비만 확률 보정 (BMI 기반 조정)
        prob_df["비만"] = (prob_df["비만"] + scaled_BMI) / 2  # 평균값 적용하여 조정

        # 🔹 pandas DataFrame으로 변환 후 Streamlit에서 표시
        prob_df = pd.DataFrame(prob_df, index=["예측 확률 (%)"])
        st.dataframe(prob_df)

        # 📌 결과 해석
        st.markdown("### 📢 건강 진단 결과")
        for disease, value in prob_df.iloc[0].items():
            if value > 75:
                st.error(f"🚨 **{disease} 위험이 매우 높습니다! 즉각적인 관리가 필요합니다.**")
            elif value > 50:
                st.warning(f"⚠️ **{disease} 위험이 높습니다. 생활습관 개선이 필요합니다.**")
            elif value > 30:
                st.info(f"ℹ️ **{disease} 위험이 중간 수준입니다. 건강 관리를 신경 써 주세요.**")
            else:
                st.success(f"✅ **{disease} 위험이 낮습니다. 건강을 유지하세요!**")

if __name__ == "__main__":
    run_eda()
