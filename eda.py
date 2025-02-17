import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go

# ✅ 모델 불러오기
try:
    model = joblib.load("classifier2_model.pkl")
except Exception as e:
    st.error(f"⚠️ 모델 로딩 실패: {e}")
    model = None


  
def run_eda():

    

    st.title("🩺 건강 예측 AI")
    st.markdown("📌 **아래 설문지를 작성하면 AI가 건강 위험도를 예측합니다.**")

    with st.form("health_form"):
        st.markdown("### 📝 **개인정보 설문**")
        col1, col2 = st.columns(2)
        with col1:
            gender = st.radio("🔹 성별", ["여성", "남성"])
            age = st.slider("🔹 나이", 10, 100, 40)
        with col2:
            height = st.number_input("🔹 키 (cm)", min_value=120, max_value=250, value=170)
            weight = st.number_input("🔹 몸무게 (kg)", min_value=30, max_value=200, value=70)

        st.markdown("---")
        st.markdown("### 💖 **건강 정보 입력**")
        col3, col4 = st.columns(2)
        with col3:
            systolic_bp = st.number_input("💓 수축기(최고) 혈압 (mmHg)", min_value=50, max_value=200, value=120)
        with col4:
            diastolic_bp = st.number_input("🩸 이완기(최저) 혈압 (mmHg)", min_value=40, max_value=150, value=80)

        st.markdown("---")
        st.markdown("### 🏃 **생활 습관 입력**")
        col5, col6, col7 = st.columns(3)
        with col5:
            smoke = st.checkbox("🚬 흡연 여부")
        with col6:
            alco = st.checkbox("🍺 음주 여부")
        with col7:
            active = st.checkbox("🏃 운동 여부")

        submit = st.form_submit_button("🔮 예측하기")

    if submit:
        try:
            # ✅ 입력 데이터 변환
            gender_value = 1 if gender == "남성" else 0
            bp_ratio = round(systolic_bp / diastolic_bp, 2) if diastolic_bp > 0 else 0
            BMI = round(weight / ((height / 100) ** 2), 2) if height > 0 else 0
            blood_pressure_diff = systolic_bp - diastolic_bp

            input_data = np.array([[ 
                gender_value, age, height, weight,
                int(smoke), int(alco), int(active), systolic_bp, diastolic_bp,
                bp_ratio, BMI, blood_pressure_diff
            ]])

            

            if model:
                if hasattr(model, "predict_proba"):
                    predicted_probs = model.predict_proba(input_data)
                else:
                    predicted_probs = model.predict(input_data)

               
                # 🔍 예측 결과 변환 (모델의 반환 형태에 맞게 처리)
                if isinstance(predicted_probs, list):
                    predicted_probs = np.array([float(arr[0, 1]) for arr in predicted_probs])
                elif isinstance(predicted_probs, np.ndarray):
                    if predicted_probs.ndim == 3:
                        predicted_probs = predicted_probs[:, 0, 1].flatten()
                    elif predicted_probs.ndim == 2:
                        predicted_probs = predicted_probs[:, 1].flatten()
                    elif predicted_probs.ndim == 1:
                        pass
                else:
                    st.error(f"⚠️ 예측 결과를 변환할 수 없습니다. 형태: {predicted_probs}")
                    return

                if len(predicted_probs) < 4:
                    st.error(f"⚠️ 모델이 4개의 질병을 예측하지 않습니다. 예측 크기: {len(predicted_probs)}")
                    return

                diseases = ["고혈압", "비만", "당뇨병", "고지혈증"]
                disease_probabilities = {diseases[i]: float(predicted_probs[i] * 100) for i in range(4)}

            else:
                st.error("⚠️ 모델이 로드되지 않아 기본값(0%)을 반환합니다.")
                disease_probabilities = {disease: 0 for disease in ["고혈압", "비만", "당뇨병", "고지혈증"]}

        except Exception as e:
            st.error(f"⚠️ 예측 오류 발생: {e}")
            return

        st.markdown("---")
        st.markdown("### 📢 **건강 예측 결과**")

        for disease, prob in disease_probabilities.items():
            safe_prob = min(1, max(0, prob / 100))  # ✅ 0~1 범위 조정
            st.metric(label=f"📊 {disease} 위험", value=f"{prob:.2f}%")
            st.progress(safe_prob)

        st.write("\n### ✅ 건강 진단 및 조치 추천 ✅")

        def show_health_risk(disease, very_high=90, high=75, moderate=50, low=35):
            prob = disease_probabilities[disease]
            if prob > very_high:
                st.error(f"🚨 {disease} 위험이 매우 높습니다! 병원 방문을 추천합니다.")
            elif prob > high:
                st.warning(f"⚠️ {disease} 위험이 높습니다. 생활습관 개선이 필요합니다.")
            elif prob > moderate:
                st.info(f"ℹ️ {disease} 위험이 중간 수준입니다. 건강 관리가 필요합니다.")
            else:
                st.success(f"✅ {disease} 위험이 낮은 편입니다. 건강한 습관을 유지하세요.")

        for disease in disease_probabilities:
            show_health_risk(disease)


        avg_values_male = {
        "몸무게 (kg)": 74, "대한민국 평균 BMI": 24.8,
        "수축기 혈압": 120, "이완기 혈압": 78,
        "고혈압 위험": 30, "당뇨병 위험": 15, "고지혈증 위험": 25
    }

        avg_values_female = {
        "몸무게 (kg)": 62, "대한민국 평균 BMI": 24.2,
        "수축기 혈압": 115, "이완기 혈압": 75,
        "고혈압 위험": 28, "당뇨병 위험": 12, "고지혈증 위험": 20
    }
    
        # ✅ 평균 비교 차트 추가 (Plotly 활용)
        st.markdown("---")  # 🔹 구분선 추가
        st.markdown("### 📊 **대한민국 평균값 vs. 유저의 결과값 비교**")
        st.info(
            f"입력한 건강 정보와 일반적인 {gender} 건강 지표를 비교합니다.\n\n"
            "- **파란색:** 대한민국 평균 수치\n"
            "- **빨간색:** 입력한 사용자 데이터\n\n"
            "이를 통해 자신의 건강 상태가 일반적인 평균과 비교해 어느 정도 차이가 있는지 시각적으로 확인할 수 있습니다."
        )

        avg_values = avg_values_male if gender == "남성" else avg_values_female
        user_values = {
            "몸무게 (kg)": weight, "사용자 BMI": BMI,
            "수축기 혈압": systolic_bp, "이완기 혈압": diastolic_bp,
            "고혈압 위험": disease_probabilities["고혈압"],
            "당뇨병 위험": disease_probabilities["당뇨병"],
            "고지혈증 위험": disease_probabilities["고지혈증"]
        }

        # ✅ Plotly 차트 생성
        fig = go.Figure()
        categories = list(avg_values.keys())

        fig.add_trace(go.Bar(
            x=categories, y=list(avg_values.values()),
            name="대한민국 평균", marker_color="blue", opacity=0.7
        ))

        fig.add_trace(go.Bar(
            x=categories, y=list(user_values.values()),
            name="유저 결과값", marker_color="red", opacity=0.7
        ))

        fig.update_layout(
            title="📊 평균값과 결과값 비교",
            xaxis_title="건강 지표",
            yaxis_title="수치",
            barmode="group",
            template="plotly_white",
            margin=dict(l=40, r=40, t=60, b=40),
            height=600  # 🔥 차트 크기 확대
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
