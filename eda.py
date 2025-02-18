
import joblib
import numpy as np
import streamlit as st
import pandas as pd
from scipy.special import expit  # 시그모이드 함수
import plotly.graph_objects as go


# ✅ BMI 계산 함수
def calculate_bmi(weight, height):
    if height > 0:
        return round(weight / ((height / 100) ** 2), 2)*100
    return 0

# ✅ 혈압 차 계산 함수
def calculate_bp_difference(systolic_bp, diastolic_bp):
    return systolic_bp - diastolic_bp

# ✅ 스케일링 적용 (흡연, 음주 영향력 확대)
def scale_binary_feature(value, scale_factor=10):
    return value * scale_factor  # 0 → 0, 1 → 10으로 확장

# ✅ 질병 확률 보정 함수 (흡연이 비만 제외, 고지혈증/고혈압/당뇨에만 영향)
def adjust_probabilities(probabilities, smoke, alco, active):
    for disease in probabilities:
        if smoke == 10:  
            if disease != "비만":  # ✅ 비만 제외하고 흡연 시 확률 증가
                probabilities[disease] += 10  
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
        BMI = calculate_bmi(weight, height)
        blood_pressure_diff = calculate_bp_difference(systolic_bp, diastolic_bp)
        bp_ratio = round(systolic_bp / diastolic_bp, 2) if diastolic_bp > 0 else 0

        # ✅ 유저 입력을 기반으로 데이터 생성
        input_data = np.array([[1 if gender == "남성" else 0, age, height, weight, 
                                smoke, alco, active, systolic_bp, diastolic_bp, 
                                bp_ratio, BMI, blood_pressure_diff]])
        
        st.write("📌 모델 입력 데이터:", input_data)

        # ✅ 모델 로드
        model = joblib.load("multioutput_classifier.pkl")

        # ✅ 예측 수행
        predicted_probs = np.array(model.predict_proba(input_data))

        # 🔹 3D 배열일 경우 2D로 변환
        if predicted_probs.ndim == 3:
            predicted_probs = predicted_probs.squeeze(axis=1)  # (4,2) 형태로 변경

        # 📌 예측 확률 결과를 데이터프레임으로 변환
        diseases = ["고혈압", "비만", "당뇨병", "고지혈증"]
        prob_df = {diseases[i]: predicted_probs[i][1] * 100 for i in range(len(diseases))}  # 양성 확률 (1) 만 출력

        # ✅ 흡연/음주/운동에 대한 확률 보정 적용 (흡연이 비만에는 영향 없음)
        prob_df = adjust_probabilities(prob_df, smoke, alco, active)

        # 🔹 pandas DataFrame으로 변환 후 Streamlit에서 표시
        prob_df = pd.DataFrame(prob_df, index=["예측 확률 (%)"])
        st.dataframe(prob_df)

        # 📌 결과 해석
        st.markdown("### 📢 건강 진단 결과")
        for disease, value in prob_df.iloc[0].items():
            if value > 85:
                st.error(f"🚨 **{disease} 위험이 매우 높습니다! 즉각적인 관리가 필요합니다.**")
            elif value > 75:
                st.warning(f"⚠️ **{disease} 위험이 높습니다. 생활습관 개선이 필요합니다.**")
            elif value > 60:
                st.info(f"ℹ️ **{disease} 위험이 중간 수준입니다. 건강 관리를 신경 써 주세요.**")
            else:
                st.success(f"✅ **{disease} 위험이 낮습니다. 건강을 유지하세요!**")




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

        st.markdown("---")
        st.markdown("### 📢 **건강 예측 결과**")
        st.write("")
        st.write("")

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="💓 고혈압 위험", value=f"{prob_df['고혈압']:.2f}%")
            st.progress(prob_df["고혈압"] / 100)
            st.metric(label="⚖️ 비만 위험", value=f"{prob_df['비만']:.2f}%")
            st.progress(prob_df["비만"] / 100)
        with col2:
            st.metric(label="🍬 당뇨병 위험", value=f"{prob_df['당뇨병']:.2f}%")
            st.progress(prob_df["당뇨병"] / 100)
            st.metric(label="🩸 고지혈증 위험", value=f"{prob_df['고지혈증']:.2f}%")
            st.progress(prob_df["고지혈증"] / 100)

        st.write("")
        st.write("")

        # 건강 진단 및 추천
        st.write("### ✅ 건강 진단 및 조치 추천 ✅")
        def show_health_risk(disease, very_high=90, high=75, moderate=50, low=35):
            prob = prob_df[disease]
            if prob > very_high:
                st.error(f"🚨 **{disease} 위험이 매우 높습니다! 즉각적인 관리가 필요합니다. 병원 방문을 추천합니다.**")
            elif prob > high:
                st.warning(f"⚠️ **{disease} 위험이 높습니다. 생활습관 개선이 필요합니다. 주기적인 건강 체크를 권장합니다.**")
            elif prob > moderate:
                st.info(f"ℹ️ **{disease} 위험이 중간 수준입니다. 생활습관 개선을 고려하세요. 운동과 식이조절이 필요할 수 있습니다.**")
            elif prob > low:
                st.success(f"✅ **{disease} 위험이 낮은 편입니다. 건강한 습관을 유지하세요.**")
            else:
                st.success(f"🎉 **{disease} 위험이 매우 낮습니다! 현재 건강 상태가 양호합니다. 건강을 꾸준히 관리하세요.**")

        show_health_risk("고혈압", 90, 70, 50, 35)
        show_health_risk("비만", 80, 50, 40, 20)
        show_health_risk("당뇨병", 70, 60, 50, 20)
        show_health_risk("고지혈증", 70, 60, 40, 25)

        # ▶️ 평균 비교 차트 (Plotly)
        # 차트에서 '나이'와 '키'는 제거하고, '사용자 BMI'를 '몸무게 (kg)' 옆에 표시
        st.markdown("---")
        st.markdown("### 📊 **평균 vs. 입력값 비교**")
        st.info(
            f"입력한 건강 정보와 일반적인 {gender} 건강 지표를 비교합니다.\n\n"
            "- **파란색:** 대한민국 평균 수치\n"
            "- **빨간색:** 입력한 사용자 데이터\n\n"
            "이를 통해 자신의 건강 상태가 일반적인 평균과 비교해 어느 정도 차이가 있는지 시각적으로 확인할 수 있습니다."
        )

        # 차트용 데이터 구성 (나이와 키 제거, BMI는 몸무게 옆에 표시)
        avg_chart = {
            "몸무게 (kg)": avg_values_male["몸무게 (kg)"] if gender=="남성" else avg_values_female["몸무게 (kg)"],
            "대한민국 평균 BMI": avg_values_male["대한민국 평균 BMI"] if gender=="남성" else avg_values_female["대한민국 평균 BMI"],
            "수축기 혈압": avg_values_male["수축기 혈압"] if gender=="남성" else avg_values_female["수축기 혈압"],
            "이완기 혈압": avg_values_male["이완기 혈압"] if gender=="남성" else avg_values_female["이완기 혈압"],
            "고혈압 위험": avg_values_male["고혈압 위험"] if gender=="남성" else avg_values_female["고혈압 위험"],
            "당뇨병 위험": avg_values_male["당뇨병 위험"] if gender=="남성" else avg_values_female["당뇨병 위험"],
            "고지혈증 위험": avg_values_male["고지혈증 위험"] if gender=="남성" else avg_values_female["고지혈증 위험"],
        }

        user_chart = {
            "몸무게 (kg)": weight,
            "사용자 BMI": BMI,
            "수축기 혈압": systolic_bp,
            "이완기 혈압": diastolic_bp,
            "고혈압 위험": prob_df["고혈압"],
            "당뇨병 위험": prob_df["당뇨병"],
            "고지혈증 위험": prob_df["고지혈증"],
        }

        categories = list(user_chart.keys())
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=categories, y=list(avg_chart.values()),
            name="대한민국 평균", marker_color="blue", opacity=0.7
        ))
        fig.add_trace(go.Bar(
            x=categories, y=list(user_chart.values()),
            name="유저 입력값", marker_color="red", opacity=0.7
        ))
        fig.update_layout(
            title="📊 평균값과 입력값 비교",
            xaxis_title="건강 지표",
            yaxis_title="수치",
            barmode="group",
            template="plotly_white",
            margin=dict(l=40, r=40, t=60, b=40),
            height=600
        )
        st.plotly_chart(fig)

        st.markdown("### 📌 **건강 지표 설명**")
        st.info(
            "- **BMI (체질량지수)**: 체중(kg)을 키(m)의 제곱으로 나눈 값으로, 비만 여부를 평가하는 지표입니다. **BMI 25 이상이면 과체중, 30 이상이면 비만**으로 간주됩니다.\n"
            "- **수축기 & 이완기 혈압**: 혈압 측정값 (높을수록 건강 위험 증가)\n"
            "- **고혈압 위험**: 혈압이 정상 범위를 초과할 경우 고혈압 위험 증가\n"
            "- **당뇨병 위험**: 혈당 수치가 높거나 생활습관 요인에 따라 당뇨병 가능성이 높아짐\n"
            "- **고지혈증 위험**: 혈중 콜레스테롤 수치가 높을 경우 혈관 질환 발생 가능성이 증가\n"
            "- **대한민국 평균값**: 한국 성인 평균 건강 지표 (참고용)\n"
        ) 
if __name__ == "__main__":
    run_eda()
