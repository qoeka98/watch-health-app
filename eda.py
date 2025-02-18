import joblib
import numpy as np
import streamlit as st
import pandas as pd

def run_eda():
    st.title(" 건강 예측 AI")
    st.markdown("📌 **건강 정보를 입력하면 AI가 질병 발생 확률을 예측합니다.**")

    # ✅ 사용자 입력 폼
    with st.form("user_input_form"):
        gender = st.radio("🔹 성별", ["여성", "남성"])
        age = st.slider("🔹 나이", 10, 100, 40)
        height = st.number_input("🔹 키 (cm)", min_value=120, max_value=250, value=170)
        weight = st.number_input("🔹 몸무게 (kg)", min_value=30, max_value=200, value=70)
        bmi_choice = st.radio("🔹 BMI 입력 방식", ["자동 계산 (키 & 몸무게 기반)", "직접 입력"])
        
        if bmi_choice == "자동 계산 (키 & 몸무게 기반)":
            BMI = round(weight / ((height / 100) ** 2), 2)
        else:
            BMI = st.number_input("🔹 BMI 직접 입력", min_value=10.0, max_value=50.0, value=24.2, step=0.1)
        
        systolic_bp = st.number_input("💓 수축기 혈압 (mmHg)", min_value=50, max_value=200, value=120)
        diastolic_bp = st.number_input("🩸 이완기 혈압 (mmHg)", min_value=40, max_value=150, value=80)
        
        smoke = 1 if st.checkbox("🚬 흡연 여부") else 0
        alco = 1 if st.checkbox("🍺 음주 여부") else 0
        active = 1 if st.checkbox("🏃 운동 여부") else 0

        # ✅ 폼 제출 버튼
        submit = st.form_submit_button("🔮 예측하기")

    if submit:
        # ✅ BMI 및 기타 계산
        bp_ratio = round(systolic_bp / diastolic_bp, 2) if diastolic_bp > 0 else 0
        blood_pressure_diff = systolic_bp - diastolic_bp

        # ✅ 유저 입력을 기반으로 데이터 생성
        input_data = np.array([[1 if gender == "남성" else 0, age, height, weight, 
                                smoke, alco, active, systolic_bp, diastolic_bp, 
                                bp_ratio, BMI, blood_pressure_diff]])
        
        st.write(f"📌 입력된 BMI: {BMI}")
        st.write(f"📌 모델 입력 데이터: {input_data}")

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

        # 🔹 pandas DataFrame으로 변환 후 Streamlit에서 표시
        prob_df = pd.DataFrame(prob_df, index=["예측 확률 (%)"])
        st.dataframe(prob_df)

        # 📌 비만 예측 확률 확인
        st.write(f"📌 비만 예측 확률: {prob_df.loc['예측 확률 (%)', '비만']}%")

        # 📌 결과 해석
        st.markdown("### 📢 건강 진단 결과")
        for disease, value in prob_df.iloc[0].items():
            if value > 75:
                st.error(f"🚨 **{disease} 위험이 매우 높습니다!! 즉각적인 관리가 필요합니다.**")
            elif value > 50:
                st.warning(f"⚠️ **{disease} 위험이 높습니다. 생활습관 개선이 필요합니다.**")
            elif value > 30:
                st.info(f"ℹ️ **{disease} 위험이 중간 수준입니다. 건강 관리를 신경 써 주세요.**")
            else:
                st.success(f"✅ **{disease} 위험이 낮습니다. 건강을 유지하세요!**")

if __name__ == "__main__":
    run_eda()
