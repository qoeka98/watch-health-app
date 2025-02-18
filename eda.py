import joblib
import numpy as np

def run_eda():

    # 모델 로드
    model = joblib.load("multioutput_classifier.pkl")

    # 모델 구조 확인
    print("모델 타입:", type(model))
    print("모델 내부 개별 분류기 타입:", [type(est) for est in model.estimators_])



    # 테스트 입력 데이터
    input_data = np.array([[1, 40, 170, 70, 1, 1, 1, 120, 80, 1.5, 24.2, 40]])

    print("입력 데이터 형태:", input_data.shape)

