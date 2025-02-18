import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go
import xgboost as xgb

# 🔹 MultiOutputClassifier 로드 (joblib 사용)
model = joblib.load("multioutput_classifier.pkl")

# 🔹 내부 XGBoost 모델 개별 로드 후 `n_classes_` 설정 추가
for i in range(len(model.estimators_)):
    booster = xgb.Booster()
    booster.load_model(f"xgb_model_{i}.json")  # JSON 파일에서 불러오기
    model.estimators_[i] = xgb.XGBClassifier()
    model.estimators_[i]._Booster = booster  # Booster 연결
    
    # 🔹 `n_classes_` 수동 설정 (XGBoost의 다중 클래스 분류 문제 해결)
    model.estimators_[i].n_classes_ = 2  # 이진 분류이므로 2로 설정

# 🔹 고혈압 위험도 계산 함수 (혈압 기반 휴리스틱 적용)
def calculate_hypertension_risk(systolic_bp, diastolic_bp, blood_pressure_diff, smoke, alco, active):
    base_risk = 10  # 기본값
    base_risk += max(0, (systolic_bp - 120) * 1.5)  # 수축기 혈압
    base_risk += max(0, (diastolic_bp - 80) * 1.2)  # 이완기 혈압
    base_risk += max(0, (blood_pressure_diff - 50) * 0.5)  # 혈압 차이
    if smoke == 0: base_risk += 10
    if alco == 0: base_risk += 10
    if active == 0: base_risk -= 10
    return min(max(base_risk, 0), 100)  # 0~100 범위 제한

# 🔹 Streamlit 앱 실행
def run_eda():
    st.title("🩺 건강 예측 AI")
    st.markdown("📌 **아래 설문지를 작성하면 AI가 건강 위험도를 예측합니다.**")

    with st.form("health_form"):
        st.markdown("### 📝 개인정보 입력")
        col1, col2 = st.columns(2)
        with col1:
            gender = st.radio("🔹 성별", ["여성", "남성"])
            age = st.slider("🔹 나이", 10, 100, 40)
        with col2:
            height = st.number_input("🔹 키 (cm)", min_value=120, max_value=250, value=170)
            weight = st.number_input("🔹 몸무게 (kg)", min_value=30, max_value=200, value=70)

        st.markdown("### 💖 건강 정보 입력")
        col3, col4 = st.columns(2)
        with col3:
            systolic_bp = st.number_input("💓 수축기(최고) 혈압 (mmHg)", min_value=50, max_value=200, value=120)
        with col4:
            diastolic_bp = st.number_input("🩸 이완기(최저) 혈압 (mmHg)", min_value=40, max_value=150, value=80)

        st.markdown("### 🏃 생활 습관 입력")
        col5, col6, col7 = st.columns(3)
        with col5:
            smoke = st.checkbox("🚬 흡연 여부")
            smoke = 0 if smoke else 1
        with col6:
            alco = st.checkbox("🍺 음주 여부")
            alco = 0 if alco else 1
        with col7:
            active = st.checkbox("🏃 운동 여부")
            active = 0 if active else 1

        submit = st.form_submit_button("🔮 예측하기")

    if submit:
        bp_ratio = round(systolic_bp / diastolic_bp, 2) if diastolic_bp > 0 else 0
        BMI = round(weight / ((height / 100) ** 2), 2)
        blood_pressure_diff = systolic_bp - diastolic_bp
        hypertension_risk = calculate_hypertension_risk(systolic_bp, diastolic_bp, blood_pressure_diff, smoke, alco, active)

        input_data = np.array([[1 if gender == "남성" else 0, age, height, weight, smoke, alco, active,
                                systolic_bp, diastolic_bp, bp_ratio, BMI, blood_pressure_diff]])

        predicted_probs = np.array(model.predict_proba(input_data))
        diseases = ["비만", "당뇨병", "고지혈증"]  # 고혈압은 따로 계산
        disease_probabilities = {diseases[i]: predicted_probs[i][0][1] * 100 for i in range(len(diseases))}
        disease_probabilities["고혈압"] = hypertension_risk

        # 📌 비만 위험도 재계산
        if BMI <= 16:
            obesity_risk = 5
        elif BMI <= 23:
            obesity_risk = ((BMI - 16) / (25 - 16)) * (50 - 5) + 5
        elif BMI <= 40:
            obesity_risk = ((BMI - 25) / (40 - 25)) * (100 - 50) + 50
        else:
            obesity_risk = 100
        disease_probabilities["비만"] = obesity_risk

        # 📌 결과 시각화
        st.markdown("### 📢 건강 예측 결과")
        col1, col2 = st.columns(2)
        for i, (disease, value) in enumerate(disease_probabilities.items()):
            with col1 if i % 2 == 0 else col2:
                st.metric(label=f"💡 {disease} 위험", value=f"{value:.2f}%")
                st.progress(value / 100)

        # 📌 건강 진단 메시지
        def show_health_risk(disease, very_high=90, high=75, moderate=50, low=35):
            prob = disease_probabilities[disease]
            if prob > very_high:
                st.error(f"🚨 **{disease} 위험이 매우 높습니다! 즉각적인 관리가 필요합니다.**")
            elif prob > high:
                st.warning(f"⚠️ **{disease} 위험이 높습니다. 생활습관 개선이 필요합니다.**")
            elif prob > moderate:
                st.info(f"ℹ️ **{disease} 위험이 중간 수준입니다. 운동과 식이조절을 고려하세요.**")
            else:
                st.success(f"✅ **{disease} 위험이 낮습니다. 건강한 습관을 유지하세요.**")

        for disease in disease_probabilities:
            show_health_risk(disease)




        
        # [8] 평균 비교 차트 (나이, 키 제외; 몸무게 옆에 사용자 BMI 표시)
        st.markdown("---")
        st.markdown("### 📊 **평균 vs. 입력값 비교**")
        st.info(
            f"입력한 건강 정보와 일반적인 {gender} 건강 지표를 비교합니다.\n\n"
            "- **파란색:** 대한민국 평균 수치\n"
            "- **빨간색:** 입력한 사용자 데이터\n\n"
            "이를 통해 자신의 건강 상태가 일반적인 평균과 비교해 어느 정도 차이가 있는지 시각적으로 확인할 수 있습니다."
        )
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
            "고혈압 위험": disease_probabilities["고혈압"],
            "당뇨병 위험": disease_probabilities["당뇨병"],
            "고지혈증 위험": disease_probabilities["고지혈증"],
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
