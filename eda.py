import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go

# 모델 불러오기
model = joblib.load("classifier2_model.pkl")

def run_app():
    st.title("🩺 건강 예측 AI")
    st.markdown("📌 아래 정보를 입력하면, 혈압, 비만, 당뇨, 고지혈증의 위험도를 예측합니다.")
    
    # 사용자 원시 입력 (9개)
    gender = st.radio("성별", ["여성", "남성"])
    age = st.number_input("나이", min_value=0, max_value=120, value=40)
    height = st.number_input("키 (cm)", min_value=100, max_value=250, value=170)
    weight = st.number_input("몸무게 (kg)", min_value=30, max_value=200, value=70)
    systolic_bp = st.number_input("최고 혈압 (mmHg)", min_value=80, max_value=250, value=120)
    diastolic_bp = st.number_input("최저 혈압 (mmHg)", min_value=40, max_value=150, value=80)
    alco = st.checkbox("음주 여부")
    smoke = st.checkbox("흡연 여부")
    active = st.checkbox("운동 여부")
    
    if st.button("예측하기"):
        # 1. 원시 입력값을 모델이 요구하는 형식으로 변환
        # 1-1. 생활습관 점수: (흡연 여부 + 음주 여부) - 운동 여부
        #    여기서 체크되면 True(1), 아니면 False(0)로 처리합니다.
        lifestyle_score = (1 if smoke else 0) + (1 if alco else 0) - (1 if active else 0)
        
        # 1-2. 혈압 차이 계산
        bp_diff = systolic_bp - diastolic_bp
        
        # 1-3. BMI 계산
        BMI = weight / ((height/100)**2)
        
        # 1-4. 성별 변환: 남성=1, 여성=0
        gender_val = 1 if gender == "남성" else 0
        
        # 1-5. 최종 7개 피처: [성별, 나이, 키, 몸무게, 혈압 차이, BMI, 생활습관 점수]
        features = np.array([[gender_val, age, height, weight, bp_diff, BMI, lifestyle_score]])
        
        # 2. 모델 예측: 모델은 7개 피처를 받아 4개 질환(혈압, 비만, 당뇨, 고지혈증)에 대한 양성 클래스 확률을 출력한다고 가정
        preds = model.predict_proba(features)
        
        # 3. 예측 결과 후처리: 모델이 반환하는 예측 결과의 형상에 따라 각 질환의 양성 확률 추출
        results = {}
        condition_names = ["혈압", "비만", "당뇨", "고지혈증"]
        # 만약 모델이 list 형태(각 조건별로 (1,2) 배열)를 반환하면:
        if isinstance(preds, list):
            for cond, pred in zip(condition_names, preds):
                # 양성 확률: pred[0,1]
                results[cond] = pred[0, 1] * 100
        else:
            # 만약 preds가 numpy array (예: (4,2))라면:
            for i, cond in enumerate(condition_names):
                results[cond] = preds[i, 1] * 100
        
        # 4. 결과 출력
        st.write("### 예측 결과")
        for cond, prob in results.items():
            st.metric(label=f"{cond} 위험도", value=f"{prob:.2f}%")
        
        # 5. 결과 시각화: 간단한 막대 그래프로 출력
        fig = go.Figure(data=[
            go.Bar(name='예측 결과', x=list(results.keys()), y=list(results.values()))
        ])
        fig.update_layout(title="예측 위험도", xaxis_title="질환", yaxis_title="위험도 (%)")
        st.plotly_chart(fig)

if __name__ == "__main__":
    run_app()
