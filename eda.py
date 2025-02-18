import joblib
import numpy as np
import streamlit as st
import pandas as pd
from scipy.special import expit  # 시그모이드 함수 (필요 시 사용)
import plotly.graph_objects as go


# ✅ BMI 계산 함수
def calculate_bmi(weight, height):
    if height > 0:
        # BMI 공식: weight / (height(m))^2  
        # 여기서는 소수점 2자리 반올림 후 100배 (원하는 방식으로 조정)
        return round(weight / ((height / 100) ** 2), 2) * 100
    return 0

# ✅ 혈압 차 계산 함수
def calculate_bp_difference(systolic_bp, diastolic_bp):
    return systolic_bp - diastolic_bp

# ✅ 스케일링 적용 (흡연, 음주, 운동의 영향력 확대)
def scale_binary_feature(value, scale_factor=10):
    return value * scale_factor  # 0 → 0, 1 → 10으로 확장

# ✅ 질병 확률 보정 함수
# - 흡연: 체크 시 모든 질병 위험 확률을 +10  
# - 음주: 체크 시 '비만'을 제외한 나머지 위험 확률을 +5  
# - 운동: 체크 시 모든 질병 위험 확률을 -2  
def adjust_probabilities(probabilities, smoke, alco, active):
    for disease in probabilities:
        if smoke == 10:
            probabilities[disease] += 10  
        if alco == 10:
            if disease != "비만":
                probabilities[disease] += 5  
        if active == 10:
            probabilities[disease] -= 2  
        # 확률은 0~100 범위로 제한
        probabilities[disease] = min(max(probabilities[disease], 0), 100)
    return probabilities

# ✅ 질병별 위험도에 따른 피드백 함수
def show_health_risk(disease, value):
    if disease == "고혈압":
        if value > 90:
            st.error(f"🚨 **고혈압 위험이 매우 심각합니다! 즉시 의료 상담이 필요합니다.**\n"
                     "💊 혈압 약 복용을 고려하고, 병원 방문을 강력 추천합니다.\n"
                     "⚠️ 나트륨 섭취를 줄이고, 저염식 식단을 유지하세요.")
        elif value > 75:
            st.warning(f"⚠️ **고혈압 위험이 높습니다. 생활습관 개선이 필요합니다.**\n"
                       "🍎 채소와 과일 섭취를 늘리고, 짠 음식은 피하세요.\n"
                       "🏃 규칙적인 운동(하루 30분 이상)을 실천하세요.")
        elif value > 60:
            st.info(f"🔶 **고혈압 위험이 다소 높습니다. 혈압 관리를 신경 쓰세요.**\n"
                    "🩸 주기적으로 혈압을 체크하고, 건강한 식습관을 유지하세요.")
        elif value > 40:
            st.success(f"✅ **고혈압 위험이 낮은 편입니다.**\n"
                       "👍 규칙적인 운동과 저염식을 지속하세요.")
        else:
            st.success(f"🎉 **고혈압 위험이 매우 낮습니다!**")
    
    elif disease == "비만":
        if value > 90:
            st.error(f"🚨 **비만 위험이 매우 심각합니다! 체중 조절이 시급합니다.**\n"
                     "⚠️ 칼로리 섭취를 제한하고, 저탄수화물 식단을 고려하세요.\n"
                     "🏃 고강도 유산소 운동(하루 1시간 이상)을 추천합니다.")
        elif value > 75:
            st.warning(f"⚠️ **비만 위험이 높습니다. 체중 감량이 필요합니다.**\n"
                       "🍎 균형 잡힌 식단과 규칙적인 운동을 병행하세요.\n"
                       "🏋️ 근력 운동도 도움이 됩니다.")
        elif value > 60:
            st.info(f"🔶 **비만 위험이 중간 수준입니다.**\n"
                    "🥗 신체 활동을 늘리고, 충분한 수분 섭취를 유지하세요.\n"
                    "🚶 하루 30분 이상 걷기를 실천해보세요.")
        elif value > 40:
            st.success(f"✅ **비만 위험이 낮은 편입니다.**\n"
                       "💪 꾸준한 운동과 건강한 식습관을 유지하세요.")
        else:
            st.success(f"🎉 **비만 위험이 매우 낮습니다!**")
    
    elif disease == "당뇨병":
        if value > 90:
            st.error(f"🚨 **당뇨병 위험이 매우 심각합니다! 즉각적인 혈당 관리가 필요합니다.**\n"
                     "🍬 단 음식 섭취를 엄격히 제한하고, 혈당 수치를 자주 체크하세요.\n"
                     "💉 의료진 상담을 받으세요.")
        elif value > 75:
            st.warning(f"⚠️ **당뇨병 위험이 높습니다.**\n"
                       "🥦 섬유질이 풍부한 음식을 늘리고, 규칙적인 운동을 병행하세요.")
        elif value > 60:
            st.info(f"🔶 **당뇨병 위험이 중간 수준입니다.**\n"
                    "🍏 정제 탄수화물 대신 잡곡을 섭취하고, 가벼운 운동을 하세요.")
        elif value > 40:
            st.success(f"✅ **당뇨병 위험이 낮습니다.**\n"
                       "🥗 균형 잡힌 식단과 운동을 유지하세요.")
        else:
            st.success(f"🎉 **당뇨병 위험이 매우 낮습니다!**")
    
    elif disease == "고지혈증":
        if value > 90:
            st.error(f"🚨 **고지혈증 위험이 매우 심각합니다! 즉각적인 콜레스테롤 관리가 필요합니다.**\n"
                     "🍟 포화지방이 많은 음식은 엄격히 제한하고, 오메가-3가 풍부한 음식을 섭취하세요.")
        elif value > 75:
            st.warning(f"⚠️ **고지혈증 위험이 높습니다.**\n"
                       "🥑 건강한 지방(아보카도, 견과류 등)을 섭취하고, 주기적인 운동을 실천하세요.")
        elif value > 60:
            st.info(f"🔶 **고지혈증 위험이 중간 수준입니다.**\n"
                    "🥗 식이섬유와 식물성 단백질 섭취를 늘리세요.\n"
                    "🚶 하루 30분 이상 걷기를 추천합니다.")
        elif value > 40:
            st.success(f"✅ **고지혈증 위험이 낮습니다.**\n"
                       "🍽️ 건강한 식습관과 꾸준한 운동을 유지하세요.")
        else:
            st.success(f"🎉 **고지혈증 위험이 매우 낮습니다!**")


def run_eda():
    st.title("🩺 건강 예측 AI")
    st.markdown("📌 **건강 정보를 입력하면 AI가 질병 발생 확률을 예측합니다.**")
    
    # 사용자 입력 폼
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
        
        submit = st.form_submit_button("🔮 예측하기")
    
    if submit:
        # BMI 및 혈압 차 계산
        BMI = calculate_bmi(weight, height)
        blood_pressure_diff = calculate_bp_difference(systolic_bp, diastolic_bp)
        bp_ratio = round(systolic_bp / diastolic_bp, 2) if diastolic_bp > 0 else 0

        # 모델 입력 데이터 생성
        input_data = np.array([[ 
            1 if gender == "남성" else 0,  # 성별: 남성=1, 여성=0
            age,
            height,
            weight,
            smoke,
            alco,
            active,
            systolic_bp,
            diastolic_bp,
            bp_ratio,
            BMI,
            blood_pressure_diff
        ]])
        
        # 모델 로드
        try:
            model = joblib.load("multioutput_classifier.pkl")
        except Exception as e:
            st.error("모델 파일을 불러오지 못했습니다. 모델 파일의 경로를 확인하세요.")
            st.stop()
        
        # 예측 수행 (예측 결과: 각 질병에 대한 [0, 1] 클래스의 확률)
        predicted_probs = np.array(model.predict_proba(input_data))
        # 만약 3차원 배열이면 squeeze
        if predicted_probs.ndim == 3:
            predicted_probs = predicted_probs.squeeze(axis=1)  # (4,2) 형태
        
        # 질병 이름 (순서는 모델 학습 순서와 일치해야 함)
        diseases = ["고혈압", "비만", "당뇨병", "고지혈증"]
        prob_dict = {}
        for i, disease in enumerate(diseases):
            # 양성 클래스(질병 존재) 확률에 100을 곱해 %로 변환
            prob_dict[disease] = predicted_probs[i, 1] * 100
        
        # 흡연/음주/운동에 따른 보정 적용
        prob_dict = adjust_probabilities(prob_dict, smoke, alco, active)
        
        # 결과 표시: 4개 컬럼에 각 질병의 확률 출력
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🩸 고혈압", f"{prob_dict['고혈압']:.2f}%")
        col2.metric("⚖️ 비만", f"{prob_dict['비만']:.2f}%")
        col3.metric("🍬 당뇨병", f"{prob_dict['당뇨병']:.2f}%")
        col4.metric("🧈 고지혈증", f"{prob_dict['고지혈증']:.2f}%")
        
        # 위험율을 보다 눈에 띄게 HTML 스타일로 표시
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center;'>💥 위험율 요약</h2>", unsafe_allow_html=True)
        for disease, value in prob_dict.items():
            st.markdown(f"<h3 style='color: red; text-align: center;'>{disease}: {value:.2f}%</h3>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        
        st.markdown("### 📢 **질병별 건강 진단 및 조치 추천**")
        # 질병별 맞춤 피드백 출력
        for disease, value in prob_dict.items():
            show_health_risk(disease, value)
        
        # -------------------------
        # 평균 비교 차트 (Plotly)
        # -------------------------
        st.markdown("---")
        st.markdown("### 📊 **평균 vs. 입력값 비교**")
        st.info(
            f"입력한 건강 정보와 일반적인 {gender}의 평균 건강 지표를 비교합니다.\n\n"
            "- **파란색:** 대한민국 평균 수치\n"
            "- **빨간색:** 입력한 사용자 데이터\n\n"
            "이를 통해 자신의 건강 상태가 평균과 비교해 어느 정도 차이가 있는지 확인할 수 있습니다."
        )
        
        # 남성과 여성 평균 데이터
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
        
        # 사용자 데이터 구성  
        # BMI는 따로 계산한 값을 사용 (비율로 표시하기 위해 100으로 나눔)
        user_chart = {
            "몸무게 (kg)": weight,
            "사용자 BMI": BMI / 100,
            "수축기 혈압": systolic_bp,
            "이완기 혈압": diastolic_bp,
            "고혈압 위험": prob_dict["고혈압"],
            "당뇨병 위험": prob_dict["당뇨병"],
            "고지혈증 위험": prob_dict["고지혈증"]
        }
        
        # 평균 데이터는 성별에 따라 선택
        avg_chart = {
            "몸무게 (kg)": avg_values_male["몸무게 (kg)"] if gender == "남성" else avg_values_female["몸무게 (kg)"],
            "대한민국 평균 BMI": avg_values_male["대한민국 평균 BMI"] if gender == "남성" else avg_values_female["대한민국 평균 BMI"],
            "수축기 혈압": avg_values_male["수축기 혈압"] if gender == "남성" else avg_values_female["수축기 혈압"],
            "이완기 혈압": avg_values_male["이완기 혈압"] if gender == "남성" else avg_values_female["이완기 혈압"],
            "고혈압 위험": avg_values_male["고혈압 위험"] if gender == "남성" else avg_values_female["고혈압 위험"],
            "당뇨병 위험": avg_values_male["당뇨병 위험"] if gender == "남성" else avg_values_female["당뇨병 위험"],
            "고지혈증 위험": avg_values_male["고지혈증 위험"] if gender == "남성" else avg_values_female["고지혈증 위험"]
        }
        
        # 범주 목록
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
            "- **BMI (체질량지수)**: 체중(kg)을 키(m)의 제곱으로 나눈 값으로, 비만 여부 평가 지표입니다. (BMI 25 이상이면 과체중, 30 이상이면 비만으로 간주)\n"
            "- **수축기 & 이완기 혈압**: 혈압 수치가 높을수록 건강 위험이 증가합니다.\n"
            "- **고혈압, 당뇨병, 고지혈증 위험**: 각 질병에 대한 예측 확률(%)로, 높을수록 위험 수준이 증가합니다.\n"
            "- **대한민국 평균값**: 한국 성인 평균 건강 지표 (참고용)"
        )

if __name__ == "__main__":
    run_eda()
