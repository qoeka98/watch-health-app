import streamlit as st
<<<<<<< HEAD
=======
    
>>>>>>> 687ea705c6b6ce2dcaaf3ef49706368ccc71ccf8
import numpy as np
import joblib

def main():
<<<<<<< HEAD
=======



>>>>>>> 687ea705c6b6ce2dcaaf3ef49706368ccc71ccf8
    # 모델 불러오기 (이미 저장한 모델)
    model = joblib.load('health_modelAI.pkl')

    # 앱 제목
    st.title('건강 예측 AI')

<<<<<<< HEAD
    # 사용자 입력을 받기 위한 텍스트 입력 (Diastolic Blood Pressure 추가)
    blood_glucose = st.number_input("혈당 수준 (mg/dL)", min_value=0, max_value=500, value=100)
    diastolic_bp = st.number_input("이완기 혈압 (mmHg)", min_value=50, max_value=150, value=80)  # Diastolic Blood Pressure 추가
=======
    # 사용자 입력을 받기 위한 텍스트 입력
    blood_glucose = st.number_input("혈당 수준 (mg/dL)", min_value=0, max_value=500, value=100)
>>>>>>> 687ea705c6b6ce2dcaaf3ef49706368ccc71ccf8
    systolic_bp = st.number_input("수축기 혈압 (mmHg)", min_value=50, max_value=200, value=120)
    heart_rate = st.number_input("심박수 (bpm)", min_value=40, max_value=200, value=70)
    body_temperature = st.number_input("체온 (°C)", min_value=35, max_value=42, value=37)
    spo2 = st.number_input("산소포화도 (SPO2, %)", min_value=80, max_value=100, value=95)

<<<<<<< HEAD
    # 예측 버튼 클릭 시 실행
    if st.button("예측하기"):
        # 사용자 입력을 배열로 변환
        input_data = np.array([[blood_glucose, diastolic_bp, systolic_bp, heart_rate, body_temperature, spo2]])

        # 모델 예측 확률 (predict_proba 사용)
        probabilities = model.predict_proba(input_data)

        # 질병 이름 목록
        diseases = ['당뇨병', '고혈압', '심장병', '폐질환']

        # 예측 확률을 사람이 보기 쉽게 퍼센트로 출력
        st.write("### 예측 결과:")
        for i, disease in enumerate(diseases):
            # 확률을 퍼센트로 변환 (probabilities[0][i]가 ndarray이므로 단일 값으로 변환)
            prob = probabilities[0][i].item() * 100  # 확률을 퍼센트로 변환
            st.write(f"{disease}: {prob:.2f}%")

if __name__=="__main__":
    main()
=======

    # 예측 버튼 클릭 시 실행
    if st.button("예측하기"):
        # 사용자 입력을 배열로 변환
        input_data = np.array([[blood_glucose, systolic_bp, heart_rate,body_temperature,spo2]])

        # 모델 예측
        result = model.predict(input_data)

        # 예측 결과 출력
        st.write(f"예측 결과: {result}")

if __name__=="__main__":
    main()
>>>>>>> 687ea705c6b6ce2dcaaf3ef49706368ccc71ccf8
