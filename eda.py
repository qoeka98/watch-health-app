
import joblib
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go


# ✅ BMI 계산 함수 (세분화된 기준 적용)
def calculate_bmi(weight, height):
    if height > 0:
        bmi = round(weight / ((height / 100) ** 2), 2)
        if bmi < 18.5:
            category = "저체중"
            risk_factor = 10  # 저체중도 건강 위험 있음
        elif 18.5 <= bmi < 23:
            category = "정상 체중"
            risk_factor = 5
        elif 23 <= bmi < 25:
            category = "과체중 경고"
            risk_factor = 20
        elif 25 <= bmi < 27:
            category = "과체중"
            risk_factor = 35
        elif 27 <= bmi < 30:
            category = "경도 비만"
            risk_factor = 50
        elif 30 <= bmi < 35:
            category = "중등도 비만"
            risk_factor = 70
        else:
            category = "고도 비만"
            risk_factor = 90
        return bmi, category, risk_factor
    return 0, "알 수 없음", 0

# ✅ 혈압 차 계산 함수
def calculate_bp_difference(systolic_bp, diastolic_bp):
    return systolic_bp - diastolic_bp

# ✅ 스케일링 적용 (흡연, 음주 영향력 확대)
def scale_binary_feature(value, scale_factor=10):
    return value * scale_factor  # 0 → 0, 1 → 10으로 확장

# ✅ 질병 확률 보정 함수 (비만 위험도 반영, UI 개선)
def adjust_probabilities(probabilities, smoke, alco, active, bmi_risk):
    for disease in probabilities:
        if disease == "비만":
            probabilities[disease] = max(probabilities[disease], bmi_risk)  # BMI에 기반한 최소 비만 위험 유지
        if smoke == 10 and disease != "비만":  
            probabilities[disease] += 10  # ✅ 흡연이 비만을 제외한 질병 위험 증가
        if alco == 10:
            probabilities[disease] += 5  # ✅ 음주 시 모든 질병 확률 증가
        if active == 10:
            probabilities[disease] -= 2  # ✅ 운동 시 모든 질병 확률 감소
        probabilities[disease] = min(max(probabilities[disease], 0), 100)  # 0~100 범위 제한
    return probabilities

def run_eda():
    st.title("🩺 건강 예측 AI")
    st.markdown("📌 **건강 정보를 입력하면 AI가 질병 발생 확률을 예측합니다.**")

    # ✅ 사용자 입력 폼
    with st.form("user_input_form"):
        gender = st.radio("🔹 성별", ["여성", "남성"])
        age = st.slider("🔹 나이", 10, 100, 40)
        height = st.number_input("🔹 키 (cm)", min_value=120, max_value=250, value=170)
        weight = st.number_input("🔹 몸무게 (kg)", min_value=30, max_value=200, value=70)
        systolic_bp = st.number_input("💓 수축기 혈압 (mmHg)", min_value=50, max_value=200, value=120)
        diastolic_bp = st.number_input("🩸 이완기 혈압 (mmHg)", min_value=40, max_value=150, value=80)

        smoke = scale_binary_feature(1 if st.checkbox("🚬 흡연 여부") else 0, scale_factor=10)
        alco = scale_binary_feature(1 if st.checkbox("🍺 음주 여부") else 0, scale_factor=10)
        active = scale_binary_feature(1 if st.checkbox("🏃 운동 여부") else 0, scale_factor=10)

        # ✅ 폼 제출 버튼
        submit = st.form_submit_button("🔮 예측하기")

    if submit:
        # ✅ BMI 및 혈압차 자동 계산
        BMI, bmi_category, bmi_risk = calculate_bmi(weight, height)
        blood_pressure_diff = calculate_bp_difference(systolic_bp, diastolic_bp)
        bp_ratio = round(systolic_bp / diastolic_bp, 2) if diastolic_bp > 0 else 0

        # ✅ 유저 입력을 기반으로 데이터 생성
        input_data = np.array([[1 if gender == "남성" else 0, age, height, weight, 
                                smoke, alco, active, systolic_bp, diastolic_bp, 
                                bp_ratio, BMI, blood_pressure_diff]])
        
        # ✅ 모델 로드
        model = joblib.load("multioutput_classifier.pkl")

        # ✅ 예측 수행
        predicted_probs = np.array(model.predict_proba(input_data))

        # 🔹 3D 배열일 경우 2D로 변환
        if predicted_probs.ndim == 3:
            predicted_probs = predicted_probs.squeeze(axis=1)

        # 📌 예측 확률 결과를 데이터프레임으로 변환
        diseases = ["고혈압", "비만", "당뇨병", "고지혈증"]
        prob_df = {diseases[i]: predicted_probs[i][1] * 100 for i in range(len(diseases))}

        # ✅ BMI 위험도를 비만 예측에 반영
        prob_df = adjust_probabilities(prob_df, smoke, alco, active, bmi_risk)

        # 🔹 pandas DataFrame으로 변환 후 Streamlit에서 표시
        prob_df = pd.DataFrame(prob_df, index=["예측 확률 (%)"])

        # ✅ BMI 정보 표시
        st.markdown(f"📌 **현재 BMI: {BMI} ({bmi_category})**")

        # ✅ 질병 발생 확률 보기 좋게 표시 (컬럼 UI 적용)
        st.markdown("### 🚑 **질병 발생 확률 (%)**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**🫀 고혈압**: `{prob_df['고혈압'].values[0]:.2f}%`")
            st.markdown(f"**🩸 고지혈증**: `{prob_df['고지혈증'].values[0]:.2f}%`")
        with col2:
            st.markdown(f"**⚖️ 비만**: `{prob_df['비만'].values[0]:.2f}%`")
            st.markdown(f"**🍭 당뇨병**: `{prob_df['당뇨병'].values[0]:.2f}%`")

        # 📌 결과 해석 (세분화된 건강 진단 추가)
        st.markdown("### 📢 **건강 진단 및 조치 추천**")
        def show_health_risk(disease, very_high=90, high=75, moderate=50, low=35):
            prob = prob_df[disease].values[0]
            if prob > very_high:
                st.error(f"🚨 **{disease} 위험이 매우 높습니다! 즉각적인 관리가 필요합니다. 병원 방문을 추천합니다.**")
            elif prob > high:
                st.warning(f"⚠️ **{disease} 위험이 높습니다. 생활습관 개선이 필요합니다. 건강 체크를 주기적으로 하세요.**")
            elif prob > moderate:
                st.info(f"ℹ️ **{disease} 위험이 중간 수준입니다. 식습관 개선과 운동을 고려하세요.**")
            elif prob > low:
                st.success(f"✅ **{disease} 위험이 낮은 편입니다. 건강한 생활을 유지하세요.**")
            else:
                st.success(f"🎉 **{disease} 위험이 매우 낮습니다! 건강을 꾸준히 유지하세요.**")

        show_health_risk("고혈압")
        show_health_risk("비만")
        show_health_risk("당뇨병")
        show_health_risk("고지혈증")




        

        # ------------------------------------------
        # 결과 출력 (건강 위험도, 시각화 등)
        # ------------------------------------------

        avg_values_male = {
        "나이": 45,
        "키 (cm)": 172,
        "몸무게 (kg)": 74,
        "수축기 혈압": 120,
        "이완기 혈압": 78,
        "고혈압 위험": 30,
        "당뇨병 위험": 15,
        "고지혈증 위험": 25,
        "대한민국 평균 BMI": 24.8
    }

        avg_values_female = {
        "나이": 45,
        "키 (cm)": 160,
        "몸무게 (kg)": 62,
        "수축기 혈압": 115,
        "이완기 혈압": 75,
        "고혈압 위험": 28,
        "당뇨병 위험": 12,
        "고지혈증 위험": 20,
        "대한민국 평균 BMI": 24.2
    }

        

         # ✅ 평균 비교 차트
        st.markdown("---")
        st.markdown("### 📊 **평균 vs. 입력값 비교**")

        # 차트 데이터 구성
        avg_chart = avg_values_male if gender == "남성" else avg_values_female
        user_chart = {
            "몸무게 (kg)": weight,
            "사용자 BMI": BMI,
            "수축기 혈압": systolic_bp,
            "이완기 혈압": diastolic_bp,
            "고혈압 위험": prob_df["고혈압"],
            "당뇨병 위험": prob_df["당뇨병"],
            "고지혈증 위험": prob_df["고지혈증"],
        }

        for category, value in user_chart.items():
            avg_value = avg_chart[category] if category in avg_chart else "N/A"
            st.markdown(f"**{category}**: 사용자 {value} / 대한민국 평균 {avg_value}")

        st.markdown("### 📌 **건강 지표 설명**")
        st.info(
            "- **BMI (체질량지수)**: 체중(kg)을 키(m)의 제곱으로 나눈 값으로, 비만 여부를 평가하는 지표입니다. **BMI 25 이상이면 과체중, 30 이상이면 비만**으로 간주됩니다.\n"
            "- **수축기 & 이완기 혈압**: 혈압 측정값 (높을수록 건강 위험 증가)\n"
            "- **고혈압 위험**: 혈압이 정상 범위를 초과할 경우 고혈압 위험 증가\n"
            "- **당뇨병 위험**: 혈당 수치가 높거나 생활습관 요인에 따라 당뇨병 가능성이 높아짐\n"
            "- **고지혈증 위험**: 혈중 콜레스테롤 수치가 높을 경우 혈관 질환 발생 가능성이 증가\n"
        )
if __name__ == "__main__":
    run_eda()
