import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go

# 모델 불러오기
model = joblib.load("classifier2_model.pkl")

def run_app():
    st.title("🩺 건강 예측 AI")
    st.markdown("📌 아래 정보를 입력하면, 혈압, 비만, 당뇨, 고지혈증의 위험도를 예측합니다.")

    # 평균값 설정 (남/여 기준)
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

    # 사용자 입력 폼 (원시 입력: 총 9개)
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
            # 만약 모델 학습 시 흡연이 0: 흡연 O, 1: 흡연 X였다면 반전 필요
            smoke_input = st.checkbox("🚬 흡연 여부")
            smoke = 0 if smoke_input else 1
        with col6:
            alco_input = st.checkbox("🍺 음주 여부")
            alco = 0 if alco_input else 1
        with col7:
            active_input = st.checkbox("🏃 운동 여부")
            active = 0 if active_input else 1
        
        st.write("-----")
        submit = st.form_submit_button("🔮 예측하기")
    
    if submit:
        # [1] 입력 전처리: 원시 입력 9개 → 7개 피처 계산
        # 계산 항목: 혈압 차이, BMI, 생활습관 점수 = (음주 + 흡연) - 운동
        bp_diff = systolic_bp - diastolic_bp
        BMI_val = weight / ((height/100)**2)
        lifestyle_score = (1 if alco_input else 0) + (1 if smoke_input else 0) - (1 if active_input else 0)
        # 성별: 남성=1, 여성=0
        gender_val = 1 if gender == "남성" else 0
        # 최종 입력 피처 (7개): [성별, 나이, 키, 몸무게, 혈압 차이, BMI, 생활습관 점수]
        features = np.array([[gender_val, age, height, weight, bp_diff, BMI_val, lifestyle_score]], dtype=float)
        
        # [2] 모델 예측: multioutput 모델에서는 각 조건별로 개별 추정기가 있음
        # XGBoost 기반 multioutput 모델은 predict_proba가 바로 작동하지 않을 수 있으므로, 각 추정기를 직접 호출하는 방법 사용
        preds = []
        if hasattr(model, "estimators_"):
            for estimator in model.estimators_:
                preds.append(estimator.predict_proba(features))
        else:
            preds = model.predict_proba(features)
        
        # [3] 예측 결과 후처리: 각 질환별 양성 확률 추출
        # 예상 출력: 각 추정기의 출력 shape은 (1,2)일 것으로 가정합니다.
        conditions = ["혈압", "비만", "당뇨", "고지혈증"]
        results = {}
        # 모델이 multioutput인 경우
        if isinstance(preds, list):
            for cond, pred in zip(conditions, preds):
                # pred의 shape이 (1,2)라고 가정: 인덱스 1이 양성 클래스 확률
                results[cond] = pred[0, 1] * 100
        else:
            # 단일 출력 (예: (4,2) 배열)
            for i, cond in enumerate(conditions):
                results[cond] = preds[i, 1] * 100
        
        # [4] 후처리: 후처리 로직은 모델의 원시 출력에 추가 보정을 적용하는 부분입니다.
        # 여기서는 예시로, '비만' 위험도는 BMI 기반 재계산을 사용합니다.
        if BMI_val <= 16:
            obesity_risk = 5
        elif BMI_val <= 25:
            obesity_risk = ((BMI_val - 16) / (25 - 16)) * (50 - 5) + 5
        elif BMI_val <= 40:
            obesity_risk = ((BMI_val - 25) / (40 - 25)) * (100 - 50) + 50
        else:
            obesity_risk = 100
        results["비만"] = obesity_risk
        
        # 당뇨 및 고지혈증의 경우, 모델 예측값이 높으면 위험이 낮다고 가정 (반전)
        for cond in ["당뇨", "고지혈증"]:
            results[cond] = 100 - results[cond]
        
        # [5] 라이프스타일 보정 적용
        # 고혈압: 흡연, 음주 시 위험 증가(+5), 운동 시 위험 감소(-10)
        # 당뇨, 고지혈증: 흡연 시 +5, 운동 시 -10
        # 비만: 운동 시 -10
        for cond in results:
            adj = results[cond]
            if cond == "고혈압" or cond == "혈압":
                if smoke_input:
                    adj += 5
                if alco_input:
                    adj += 5
                if active_input:
                    adj -= 10
            elif cond in ["당뇨", "고지혈증"]:
                if smoke_input:
                    adj += 5
                if active_input:
                    adj -= 10
            elif cond == "비만":
                if active_input:
                    adj -= 10
            results[cond] = min(max(adj, 0), 100)
        
        # [6] 나이 보정: 기준 나이 50세, 70세 이상은 70세 기준
        effective_age = age if age <= 70 else 70
        for cond in results:
            if cond == "고혈압" or cond == "혈압":
                results[cond] = min(max(results[cond] + 0.5*(effective_age - 50), 0), 100)
            else:
                results[cond] = min(max(results[cond] + (effective_age - 50), 0), 100)
        
        # [7] 최종 결과 출력 및 시각화
        st.write("### 예측 결과")
        for cond, prob in results.items():
            st.metric(label=f"{cond} 위험도", value=f"{prob:.2f}%")
        
        fig = go.Figure(data=[go.Bar(name='예측 결과', x=list(results.keys()), y=list(results.values()))])
        fig.update_layout(title="예측 위험도", xaxis_title="질환", yaxis_title="위험도 (%)")
        st.plotly_chart(fig)

if __name__ == "__main__":
    run_app()
