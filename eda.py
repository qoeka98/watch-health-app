import joblib
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ✅ AI 모델 로드
@st.cache_resource
def load_model():
    return joblib.load("regressor_xg")

model = load_model()

# ✅ 나이에 따른 가중치 적용 함수
def adjust_by_age(age, probabilities):
    age_factors = {"고혈압": 0, "비만": 0, "당뇨병": 0, "고지혈증": 0}
    
    if age < 30:
        age_factors = {"고혈압": 0, "비만": 5, "당뇨병": 0, "고지혈증": 0}
    elif age < 40:
        age_factors = {"고혈압": 5, "비만": 10, "당뇨병": 5, "고지혈증": 5}
    elif age < 50:
        age_factors = {"고혈압": 10, "비만": 10, "당뇨병": 10, "고지혈증": 6}
    elif age < 60:
        age_factors = {"고혈압": 25, "비만": 5, "당뇨병": 30, "고지혈증": 10}
    else:
        age_factors = {"고혈압": 25, "비만": 5, "당뇨병": 30, "고지혈증": 10}
    
    for disease in probabilities:
        probabilities[disease] += age_factors[disease]
        probabilities[disease] = min(probabilities[disease], 100)
    
    return probabilities

# ✅ 위험 수준 및 건강 조치 반환 함수
def get_health_status(probability):
    if probability <= 20:
        return "🟢 매우 안전", "✅ 건강 유지!", "위험이 거의 없습니다. 현재 건강을 잘 유지하세요!"
    elif probability <= 40:
        return "🟢 안전", "👍 건강 양호", "위험이 낮습니다. 균형 잡힌 식사를 유지하세요."
    elif probability <= 60:
        return "🟡 주의", "⚠️ 주의 필요", "위험이 증가 중입니다. 생활 습관 개선이 필요합니다."
    elif probability <= 80:
        return "🟠 위험", "🚨 건강 경고!", "위험이 높습니다. 정기 검진과 건강 관리가 필요합니다."
    else:
        return "🔴 위급", "⛔ 즉시 조치 필요!", "위험이 매우 높습니다! 병원 진료를 권장합니다."

def summarize_health(prob_dict):
    avg_risk = np.mean(list(prob_dict.values()))
    
    if avg_risk <= 20:
        return "✅ 전반적으로 건강 상태가 양호합니다! 좋은 생활 습관을 계속 유지하세요."
    elif avg_risk <= 40:
        return "👍 현재 건강 상태는 안정적입니다. 하지만 꾸준한 건강 관리가 필요합니다."
    elif avg_risk <= 60:
        return "⚠️ 건강 상태에 주의가 필요합니다. 생활 습관 개선을 고려해보세요."
    elif avg_risk <= 80:
        return "🚨 건강 위험 수준이 높아지고 있습니다. 적극적인 건강 관리가 필요합니다!"
    else:
        return "⛔ 건강 위험이 매우 높습니다! 즉시 의료 전문가와 상담하세요."

def run_eda():
    st.title("🩺 건강 예측 AI")
    st.markdown("📌 **건강 정보를 입력하면 AI가 질병 발생 확률을 예측합니다.**")

    with st.form("user_input_form"):
        gender = st.radio("🔹 성별", ["여성", "남성"])
        age = st.slider("🔹 나이", 10, 100, 40)
        height = st.number_input("🔹 키 (cm)", min_value=120, max_value=250, value=170)
        weight = st.number_input("🔹 몸무게 (kg)", min_value=30, max_value=200, value=70)
        systolic_bp = st.number_input("💓 수축기 혈압 (mmHg)", min_value=50, max_value=200, value=120)
        diastolic_bp = st.number_input("🩸 이완기 혈압 (mmHg)", min_value=40, max_value=150, value=80)

        smoke = 1 if st.checkbox("🚬 흡연 여부") else 0
        alco = 1 if st.checkbox("🍺 음주 여부") else 0

        submit = st.form_submit_button("🔮 예측하기")




    if submit:
        input_data = np.array([[ 
            systolic_bp, diastolic_bp, weight, height, smoke, alco
        ]])

        predicted_probs = model.predict(input_data)
        predicted_probs = np.clip(np.round(predicted_probs, 2), 0, 100)

        diseases = ["고혈압", "비만", "당뇨병", "고지혈증"]
        prob_dict = {diseases[i]: predicted_probs[0, i] for i in range(len(diseases))}
        prob_dict = adjust_by_age(age, prob_dict)

        
        st.markdown("## 🏥 건강 종합 진단")
        st.info(summarize_health(prob_dict))

        col1, col2 = st.columns(2)

        for i, disease in enumerate(diseases):
            status, status_text, advice = get_health_status(prob_dict[disease])

            with col1 if i % 2 == 0 else col2:
                st.subheader(f"📌 {disease}")
                st.metric(label=f"위험 확률", value=f"{prob_dict[disease]:.2f}%", delta=status)
                st.progress(int(prob_dict[disease]))
                
                if "🟢" in status:
                    st.success(f"💡 {advice}")
                elif "🟡" in status:
                    st.warning(f"💡 {advice}")
                else:
                    st.error(f"💡 {advice}")

        
        st.info(
    """
    **당뇨병 🩸**  
    혈당이 정상보다 높아지는 질환으로, 인슐린 기능이 저하되어 발생.  
    관리하지 않으면 신장, 눈, 신경 등에 합병증 위험 증가.  

    **비만 ⚖️**  
    체지방이 과도하게 축적된 상태로, 각종 성인병(당뇨, 고혈압, 심장병) 위험을 높임.  
    원인은 주로 과식, 운동 부족, 대사 문제.  

    **고지혈증 🥓**  
    혈액 내 콜레스테롤과 중성지방 수치가 높아져 혈관이 막힐 위험 증가.  
    심근경색, 뇌졸중 원인이 될 수 있음.  

    **고혈압 💓**  
    혈관 속 혈압이 정상보다 높은 상태로, 심장과 혈관에 부담을 주어  
    뇌졸중과 심장병 위험을 높임.  
    나트륨 과다 섭취와 스트레스가 주요 원인.  
    """
)
        

## ------------------------------------------------------------------------
                
                # ✅ 대한민국 평균 데이터
        avg_values = {
            "남성": {"몸무게": 74, "BMI": 24.8, "수축기 혈압": 120, "이완기 혈압": 78, "고혈압": 30, "당뇨병": 15, "고지혈증": 25},
            "여성": {"몸무게": 62, "BMI": 24.2, "수축기 혈압": 115, "이완기 혈압": 75, "고혈압": 28, "당뇨병": 12, "고지혈증": 20}
        }
        avg_data = avg_values[gender]

        # ✅ 사용자 BMI 계산 (체중 / 키(m)^2)
        BMI = round(weight / ((height / 100) ** 2), 2)

        # ✅ 사용자 입력값 정리
        user_data = {
            "몸무게 (kg)": weight,
            "사용자 BMI": BMI,
            "수축기 혈압": systolic_bp,
            "이완기 혈압": diastolic_bp,
            "고혈압 위험": prob_dict["고혈압"],
            "당뇨병 위험": prob_dict["당뇨병"],
            "고지혈증 위험": prob_dict["고지혈증"]
        }

        avg_chart = {
            "몸무게 (kg)": avg_data["몸무게"],
            "대한민국 평균 BMI": avg_data["BMI"],
            "수축기 혈압": avg_data["수축기 혈압"],
            "이완기 혈압": avg_data["이완기 혈압"],
            "고혈압 위험": avg_data["고혈압"],
            "당뇨병 위험": avg_data["당뇨병"],
            "고지혈증 위험": avg_data["고지혈증"]
        }

        # ✅ 평균 비교 차트 (Plotly)
        st.markdown("---")
        st.markdown("### 📊 **평균 vs. 입력값 비교**")
        st.info(
            f"입력한 건강 정보와 일반적인 {gender}의 평균 건강 지표를 비교합니다.\n\n"
            "- **🔵 파란색:** 대한민국 평균 수치\n"
            "- **🔴 빨간색:** 입력한 사용자 데이터\n\n"
            "이를 통해 자신의 건강 상태가 평균과 비교해 어느 정도 차이가 있는지 확인할 수 있습니다."
        )

        categories = list(user_data.keys())
        fig = go.Figure()
        fig.add_trace(go.Bar(x=categories, y=list(avg_chart.values()), name="대한민국 평균", marker_color="blue", opacity=0.7))
        fig.add_trace(go.Bar(x=categories, y=list(user_data.values()), name="유저 입력값", marker_color="red", opacity=0.7))
        fig.update_layout(
            title="📊 평균값과 입력값 비교",
            xaxis_title="건강 지표", yaxis_title="수치",
            barmode="group", template="plotly_white",
            margin=dict(l=40, r=40, t=60, b=40), height=600
        )
        st.plotly_chart(fig)

        # ✅ 건강 지표 설명
        st.markdown("### 📌 **건강 지표 설명**")
        st.info(
            "- **BMI (체질량지수)**: 체중(kg)을 키(m)의 제곱으로 나눈 값. (BMI 25 이상 = 과체중, 30 이상 = 비만)\n"
            "- **수축기 & 이완기 혈압**: 혈압 수치가 높을수록 건강 위험 증가.\n"
            "- **고혈압, 당뇨병, 고지혈증 위험**: 각 질병에 대한 AI 예측 확률(%)로, 높을수록 위험 수준 증가.\n"
            "- **대한민국 평균값**: 한국 성인 평균 건강 지표 (참고용)."
        )
