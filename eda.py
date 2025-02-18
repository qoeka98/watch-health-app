import joblib
import numpy as np
import streamlit as st

def run_eda():
    # 모델 로드
    model = joblib.load("multioutput_classifier.pkl")

    # 모델 구조 확인
    st.write("📌 모델 타입:", type(model))
    st.write("📌 모델 내부 개별 분류기 타입:", [type(est) for est in model.estimators_])

    # 테스트 입력 데이터
    input_data = np.array([[1, 40, 170, 70, 1, 1, 1, 120, 80, 1.5, 24.2, 40]])
    st.write("📌 입력 데이터 형태:", input_data.shape)

    # 예측 확률 계산
    predicted_probs = np.array(model.predict_proba(input_data))

    # 📌 예측 확률 결과 형태 확인
    st.write("📌 예측 확률 원본 형태:", predicted_probs.shape)

    # 🔹 3D 배열일 경우 2D로 변환
    if predicted_probs.ndim == 3:
        predicted_probs = predicted_probs.squeeze(axis=1)  # 차원 축소하여 (4,2) 형태로 변경

    # 🔹 2D 배열인지 확인 후 출력
    st.write("📌 변환된 예측 확률 형태:", predicted_probs.shape)

    # 📌 예측 확률 값 출력 (데이터프레임 형태로 보기 쉽게)
    diseases = ["고혈압", "비만", "당뇨병", "고지혈증"]
    prob_df = {diseases[i]: predicted_probs[i] for i in range(len(diseases))}
    
    # 🔹 pandas DataFrame으로 변환 후 Streamlit에서 표시
    import pandas as pd
    prob_df = pd.DataFrame(prob_df, index=["음성 확률 (0)", "양성 확률 (1)"])
    st.dataframe(prob_df)

if __name__ == "__main__":
    run_eda()
