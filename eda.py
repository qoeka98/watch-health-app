import joblib
import numpy as np
import streamlit as st
from scipy.special import expit  # 시그모이드 함수

def run_eda():

    # 모델 로드
    model = joblib.load("multioutput_classifier.pkl")

    # 모델 구조 확인
    print("모델 타입:", type(model))
    print("모델 내부 개별 분류기 타입:", [type(est) for est in model.estimators_])

    # 테스트 입력 데이터
    input_data = np.array([[1, 40, 170, 70, 1, 1, 1, 120, 80, 1.5, 24.2, 40]])

    print("입력 데이터 형태:", input_data.shape)

    predicted_probs = model.predict_proba(input_data)
    predicted_probs = np.array(predicted_probs)

    st.write("📌 예측 확률 결과 형태:", predicted_probs.shape)
    st.write("📌 예측 확률 값:", predicted_probs)

    