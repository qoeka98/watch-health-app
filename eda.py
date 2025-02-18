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

    print("📌 예측 확률 결과 형태:", predicted_probs.shape)
    print("📌 예측 확률 값:", predicted_probs)

    # `predict_proba()` 결과 변환 (두 번째 확률값 사용)
    disease_probabilities = {}
    diseases = ["고혈압", "비만", "당뇨병", "고지혈증"]

    for i, disease in enumerate(diseases):
        if predicted_probs.ndim == 3:
            # 3차원 배열일 경우 [1] 확률값을 가져오기
            disease_probabilities[disease] = predicted_probs[i][0][1] * 100
        elif predicted_probs.ndim == 2:
            # 2차원 배열일 경우 [1] 확률값을 가져오기
            disease_probabilities[disease] = predicted_probs[i][1] * 100
        else:
            disease_probabilities[disease] = 0  # 오류 방지

    # NaN 값 방지
    for disease in disease_probabilities:
        disease_probabilities[disease] = np.nan_to_num(disease_probabilities[disease], nan=0.0)

    print("📢 수정된 예측 확률:", disease_probabilities)

    for disease in disease_probabilities:
        # 0~100 범위 제한
        disease_probabilities[disease] = min(max(disease_probabilities[disease], 0), 100)

        # 0~1 범위로 변환
        progress_value = disease_probabilities[disease] / 100.0

        # NaN 값 방지
        if np.isnan(progress_value) or progress_value is None or np.isinf(progress_value):
            progress_value = 0.0

        # 0~1 범위 초과 방지
        progress_value = min(max(progress_value, 0.0), 1.0)

        print(f"✅ {disease} Progress Bar 값: {progress_value}")  # 디버깅
        st.progress(progress_value)

    # ✅ **보정된 확률값을 `sigmoid` 함수로 조정**
    def sigmoid_scaling(x):
        return expit((x - 0.5) * 6.5)  # 스케일 조정

    # 예측 확률 보정
    for disease in disease_probabilities:
        disease_probabilities[disease] = sigmoid_scaling(disease_probabilities[disease] / 100.0) * 100

    print("📌 보정된 예측 확률:", disease_probabilities)

    # ✅ **Min-Max Scaling 적용**
    def min_max_scale(value, min_val=0.01, max_val=0.6):  
        return ((value - min_val) / (max_val - min_val)) * 100

    for disease, value in disease_probabilities.items():
        disease_probabilities[disease] = min_max_scale(value)

    print("📌 Min-Max Scaling 적용 후:", disease_probabilities)
