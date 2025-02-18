import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go

# 모델 불러오기
model = joblib.load("classifier2_model.pkl")

def run_eda():
    st.title("🩺 건강 예측 AI")
    st.markdown("📌 아래 정보를 입력하면 혈압, 비만, 당뇨, 고지혈증의 위험도를 예측합니다.")
    
    # -------------------------------
    # 1. 평균값 설정 (남/여 기준)
    # -------------------------------
    avg_values_male = {
        "나이": 45,
        "키 (cm)": 172,
        "몸무게 (kg)": 74,
        "최고혈압": 120,
        "최저혈압": 78,
        "고혈압 위험": 30,
        "당뇨 위험": 15,
        "고지혈증 위험": 25,
        "대한민국 평균 BMI": 24.8
    }
    avg_values_female = {
        "나이": 45,
        "키 (cm)": 160,
        "몸무게 (kg)": 62,
        "최고혈압": 115,
        "최저혈압": 75,
        "고혈압 위험": 28,
        "당뇨 위험": 12,
        "고지혈증 위험": 20,
        "대한민국 평균 BMI": 24.2
    }
    
    # -------------------------------
    # 2. 사용자 입력 폼 (원시 입력: 총 9개)
    # -------------------------------
    with st.form("health_form"):
        st.markdown("### 📝 개인정보 입력")
        st.info("아래 정보를 입력해주세요.")
        col1, col2 = st.columns(2)
        with col1:
            gender = st.radio("성별", ["여성", "남성"])
            age = st.slider("나이", 10, 100, 40)
        with col2:
            height = st.number_input("키 (cm)", min_value=120, max_value=250, value=170)
            weight = st.number_input("몸무게 (kg)", min_value=30, max_value=200, value=70)
        
        st.markdown("---")
        st.markdown("### 💖 건강 관련 정보")
        col3, col4 = st.columns(2)
        with col3:
            systolic_bp = st.number_input("최고 혈압 (mmHg)", min_value=50, max_value=200, value=120)
        with col4:
            diastolic_bp = st.number_input("최저 혈압 (mmHg)", min_value=40, max_value=150, value=80)
        
        st.markdown("---")
        st.markdown("### 🏃 생활 습관 입력")
        col5, col6, col7 = st.columns(3)
        with col5:
            alco = st.checkbox("🍺 음주 여부")
        with col6:
            smoke = st.checkbox("🚬 흡연 여부")
        with col7:
            active = st.checkbox("🏃 운동 여부")
        
        st.write("-----")
        submit = st.form_submit_button("🔮 예측하기")
    
    # -------------------------------
    # 3. 입력 전처리: 9개 원시 입력 → 7개 피처 계산
    # -------------------------------
    if submit:
        # 원시 입력값:
        # 성별: '남성' -> 1, '여성' -> 0
        gender_val = 1 if gender == "남성" else 0
        # 나이, 키, 몸무게는 그대로 사용
        # 혈압 차이 = 최고 혈압 - 최저 혈압
        bp_diff = systolic_bp - diastolic_bp
        # BMI 계산
        BMI = round(weight / ((height/100)**2), 2)
        # 생활습관 점수 = (음주 여부 + 흡연 여부) - 운동 여부  
        # (체크되면 True=1, 아니면 False=0)
        lifestyle_score = (1 if alco else 0) + (1 if smoke else 0) - (1 if active else 0)
        
        # 최종 7개 피처 배열 (1행, 7열)
        features = np.array([[gender_val, age, height, weight, bp_diff, BMI, lifestyle_score]])
        
        # -------------------------------
        # 4. 모델 예측 (예: 4개 질환: 혈압, 비만, 당뇨, 고지혈증)
        # -------------------------------
        preds = model.predict_proba(features)
        # 예를 들어, 모델이 각 질환별로 (행렬 또는 리스트) 확률을 반환한다고 가정
        # (예시에서는 결과를 리스트 혹은 numpy array로 받아 처리)
        results = {}
        conditions = ["혈압", "비만", "당뇨", "고지혈증"]
        if isinstance(preds, list):
            for cond, pred in zip(conditions, preds):
                # 양성 확률: pred[0,1]
                results[cond] = pred[0, 1] * 100
        else:
            # preds가 numpy array인 경우 (예: (4,2) 배열)
            for i, cond in enumerate(conditions):
                results[cond] = preds[i, 1] * 100
        
        # -------------------------------
        # 5. 후처리 (원시 모델 출력 기반)
        # (후처리 로직은 사용자의 도메인 지식에 따라 조정 필요)
        # 여기서는 예시로 결과를 그대로 사용합니다.
        # (추가 보정 로직이 있다면 이 부분에 추가합니다.)
        final_results = results  # 후처리 로직을 추가할 수 있음
        
        # -------------------------------
        # 6. 결과 출력 및 시각화
        # -------------------------------
        st.write("### 예측 결과")
        for cond, prob in final_results.items():
            st.metric(label=f"{cond} 위험도", value=f"{prob:.2f}%")
        
        fig = go.Figure(data=[go.Bar(name='예측 결과', x=list(final_results.keys()), y=list(final_results.values()))])
        fig.update_layout(title="예측 위험도", xaxis_title="질환", yaxis_title="위험도 (%)")
        st.plotly_chart(fig)

if __name__ == "__main__":
    run_eda()
