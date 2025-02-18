import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go

# 모델 불러오기
model = joblib.load("classifier2_model.pkl")

def run_eda():
    st.title("🩺 건강 예측 AI")
    st.markdown("📌 아래 정보를 입력하면 혈압, 비만, 당뇨, 고지혈증의 위험도를 예측합니다.")
    
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
        # 모델 학습 시, 만약 흡연, 음주, 운동의 인코딩이 반대로 되었다면(0: 있음, 1: 없음)
        with col5:
            smoke_input = st.checkbox("🚬 흡연 여부")   # True이면 체크됨
            # 체크되면 실제로 흡연했다고 가정 → 모델 입력은 0 (0: 흡연 O, 1: 흡연 X)
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
        # [1] 입력 전처리
        bp_diff = systolic_bp - diastolic_bp
        BMI_val = weight / ((height/100)**2)
        # 생활습관 점수 = (음주 여부 + 흡연 여부) - 운동 여부
        # 여기서는 원래 체크 여부(음주, 흡연, 운동)가 필요하므로 사용합니다.
        lifestyle_score = (1 if alco_input else 0) + (1 if smoke_input else 0) - (1 if active_input else 0)
        gender_val = 1 if gender == "남성" else 0
        
        # 최종 7개 피처 배열 (dtype 명시적으로 float32)
        features = np.array([[gender_val, age, height, weight, bp_diff, BMI_val, lifestyle_score]], dtype=np.float32)
        
        # [2] 모델 예측 (multioutput일 경우 각 추정기 호출)
        preds = []
        if hasattr(model, "estimators_"):
            for estimator in model.estimators_:
                preds.append(estimator.predict_proba(features))
        else:
            preds = model.predict_proba(features)
        
        # [3] 모델 예측 결과 후처리: 각 질환별 양성 확률 추출
        conditions = ["혈압", "비만", "당뇨", "고지혈증"]
        results = {}
        if isinstance(preds, list):
            for cond, pred in zip(conditions, preds):
                # pred의 shape이 (1,2)라고 가정: index 1은 양성 클래스 확률
                results[cond] = pred[0, 1] * 100
        else:
            for i, cond in enumerate(conditions):
                results[cond] = preds[i, 1] * 100
        
        # [4] 후처리 보정: 비만 위험도 재계산 (BMI 기반)
        if BMI_val <= 16:
            obesity_risk = 5
        elif BMI_val <= 25:
            obesity_risk = ((BMI_val - 16) / (25 - 16)) * (50 - 5) + 5
        elif BMI_val <= 40:
            obesity_risk = ((BMI_val - 25) / (40 - 25)) * (100 - 50) + 50
        else:
            obesity_risk = 100
        results["비만"] = obesity_risk
        
        # [5] 당뇨 및 고지혈증 위험 반전 (모델 예측값이 높으면 실제 위험은 낮다고 가정)
        for cond in ["당뇨", "고지혈증"]:
            results[cond] = 100 - results[cond]
        # (혈압은 반전하지 않음)
        
        # [6] 라이프스타일 보정 적용
        # - 혈압: 흡연 시 +5, 음주 시 +5, 운동 시 -10
        # - 당뇨, 고지혈증: 흡연 시 +5, 운동 시 -10 (음주 효과 없음)
        # - 비만: 운동 시 -10
        for cond in results:
            adj = results[cond]
            if cond == "혈압":
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
        
        # [7] 나이 보정 적용 (기준 나이 50세, 70세 이상은 70세 기준)
        effective_age = age if age <= 70 else 70
        for cond in results:
            if cond == "혈압":
                results[cond] = min(max(results[cond] + 0.5 * (effective_age - 50), 0), 100)
            else:
                results[cond] = min(max(results[cond] + (effective_age - 50), 0), 100)
        
        # [8] 최종 결과 출력 및 시각화
        st.write("### 예측 결과")
        for cond, prob in results.items():
            st.metric(label=f"{cond} 위험도", value=f"{prob:.2f}%")
        
        fig = go.Figure(data=[go.Bar(name='예측 결과', x=list(results.keys()), y=list(results.values()))])
        fig.update_layout(title="예측 위험도", xaxis_title="질환", yaxis_title="위험도 (%)")
        st.plotly_chart(fig)

if __name__ == "__main__":
    run_eda()
