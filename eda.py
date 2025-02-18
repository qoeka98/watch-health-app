import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go

# 모델 불러오기
model = joblib.load("classifier2_model.pkl")

def calculate_hypertension_risk(systolic_bp, diastolic_bp, blood_pressure_diff, smoke, alco, active):
    """
    고혈압 위험도를 혈압 수치와 생활 습관을 반영하여 계산하는 함수
    """
    if systolic_bp < 120 and diastolic_bp < 80:
        base_risk = 5  # 정상 혈압일 경우 낮은 위험도
    elif systolic_bp >= 140 or diastolic_bp >= 90:
        base_risk = 80
    elif systolic_bp >= 130 or diastolic_bp >= 85:
        base_risk = 60
    elif systolic_bp >= 120 or diastolic_bp >= 80:
        base_risk = 40
    else:
        base_risk = 20
    
    # 혈압 차이에 따른 보정
    if blood_pressure_diff >= 60:
        base_risk += 15
    elif blood_pressure_diff >= 50:
        base_risk += 10
    elif blood_pressure_diff >= 40:
        base_risk += 5
    
    # 생활 습관 보정
    if smoke == 0:
        base_risk += 10
    if alco == 0:
        base_risk += 10
    if active == 0:
        base_risk -= 10
    
    return min(max(base_risk, 0), 100)

def run_eda():
    st.title("🩺 건강 예측 AI")
    st.markdown("📌 **아래 설문지를 작성하면 AI가 건강 위험도를 예측합니다.**")
    
    # 사용자 입력 폼
    with st.form("health_form"):
        st.markdown("### 📝 개인정보 입력")
        col1, col2 = st.columns(2)
        with col1:
            gender = st.radio("🔹 성별", ["여성", "남성"])
            age = st.slider("🔹 나이", 10, 100, 40)
        with col2:
            height = st.number_input("🔹 키 (cm)", min_value=120, max_value=250, value=170)
            weight = st.number_input("🔹 몸무게 (kg)", min_value=30, max_value=200, value=70)
        
        st.markdown("---")
        st.markdown("### 💖 건강 정보 입력")
        col3, col4 = st.columns(2)
        with col3:
            systolic_bp = st.number_input("💓 최고 혈압 (mmHg)", min_value=50, max_value=200, value=120)
        with col4:
            diastolic_bp = st.number_input("🩸 최저 혈압 (mmHg)", min_value=40, max_value=150, value=80)
        
        st.markdown("---")
        st.markdown("### 🏃 생활 습관 입력")
        col5, col6, col7 = st.columns(3)
        with col5:
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
        # BMI 계산
        BMI = round(weight / ((height / 100) ** 2), 2)
        
        # 혈압 차이 계산
        blood_pressure_diff = systolic_bp - diastolic_bp
        
        # 고혈압 위험도 계산 (혈압 및 혈압 차이 기반)
        hypertension_risk = calculate_hypertension_risk(systolic_bp, diastolic_bp, blood_pressure_diff, smoke, alco, active)
        
        # 결과 출력
        st.markdown("---")
        st.markdown("### 📢 **건강 예측 결과**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="💓 고혈압 위험", value=f"{hypertension_risk:.2f}%")
            st.progress(hypertension_risk / 100)
        with col2:
            st.metric(label="⚖️ BMI", value=f"{BMI:.2f}")
            st.progress(min(BMI / 40, 1))
        
        st.markdown("### 건강 진단 및 조치 추천 ")
        if hypertension_risk >= 90:
            st.error("🚨 **고혈압 위험이 매우 높습니다! 즉각적인 병원 방문을 추천합니다.**")
        elif hypertension_risk >= 70:
            st.warning("⚠️ **고혈압 위험이 높습니다. 정기적인 건강 체크와 생활습관 개선이 필요합니다.**")
        elif hypertension_risk >= 40:
            st.info("ℹ️ **고혈압 위험이 중간 수준입니다. 운동과 건강한 식습관을 고려하세요.**")
        else:
            st.success("🎉 **고혈압 위험이 낮습니다! 현재 건강한 상태를 유지하세요.**")
        
if __name__ == "__main__":
    run_eda()
