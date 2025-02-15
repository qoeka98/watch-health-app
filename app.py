import streamlit as st
import numpy as np
import joblib

def main():
    # ✅ 모델 불러오기 (저장된 모델)
    model = joblib.load("health_modelAI.pkl")

    # ✅ 웹 앱 제목
    st.title("🩺 건강 예측 AI")

    st.write("🔍 아래 정보를 입력하면 AI가 건강 위험도를 예측합니다.")

    # ✅ 사용자 입력을 받기 위한 UI 요소
    gender = st.radio("성별", ["여성", "남성"])
    height = st.number_input("키 (cm)", min_value=120, max_value=250, value=170)
    weight = st.number_input("몸무게 (kg)", min_value=30, max_value=200, value=70)
    systolic_bp = st.number_input("수축기 혈압 (mmHg)", min_value=50, max_value=200, value=120)
    diastolic_bp = st.number_input("이완기 혈압 (mmHg)", min_value=40, max_value=150, value=80)
    smoke = st.radio("흡연 여부", ["비흡연", "흡연"])
    alco = st.radio("음주 여부", ["비음주", "음주"])
    active = st.radio("운동 여부", ["운동 안함", "운동 함"])

    # ✅ 혈압 차이 자동 계산
    blood_pressure_diff = systolic_bp - diastolic_bp

    # ✅ 입력 데이터 변환
    input_data = np.array([[
        1 if gender == "남성" else 0,  # 성별 변환 (남성=1, 여성=0)
        height, weight, systolic_bp, diastolic_bp,
        1 if smoke == "흡연" else 0,  # 흡연 (1: 흡연, 0: 비흡연)
        1 if alco == "음주" else 0,   # 음주 (1: 음주, 0: 비음주)
        1 if active == "운동 함" else 0,  # 운동 여부 (1: 함, 0: 안함)
        blood_pressure_diff  # 혈압 차이
    ]])

    # ✅ 예측 버튼 클릭 시 실행
    if st.button("🔮 예측하기"):
        # AI 모델 예측 (확률 기반)
        predicted_probs = model.predict_proba(input_data)

        # ✅ 질병 이름 목록
        diseases = ["고혈압", "비만", "당뇨병", "고지혈증"]

        # ✅ 예측 결과 출력
        st.write("### 📢 건강 예측 결과:")
        for i, disease in enumerate(diseases):
            probability = predicted_probs[i][0][1] * 100  # 확률을 % 변환
            st.write(f"✅ **{disease} 확률:** {probability:.2f}%")

        st.write("💡 건강을 유지하려면 정기적인 검진과 생활 습관 개선이 필요합니다! 🩺")

if __name__ == "__main__":
    main()
