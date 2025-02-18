import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go

# 모델 불러오기
model = joblib.load("classifier2_model.pkl")

def run_eda():
    st.title("🩺 건강 예측 AI")
    st.markdown("📌 **아래 설문지를 작성하면 AI가 건강 위험도를 예측합니다.**")
    
    # 평균값 설정 (남/여 기준)
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
    
    # 사용자 입력 폼
    with st.form("health_form"):
        st.markdown("### 📝 **개인정보 설문**")
        st.info("아래 정보를 입력해주세요. (실제 값이 아닐 경우 예측 정확도가 떨어질 수 있습니다.)")
        st.write("")
        st.write("")
        
        col1, col2 = st.columns(2)
        with col1:
            gender = st.radio("🔹 성별", ["여성", "남성"])
            age = st.slider("🔹 나이", 10, 100, 40)
        with col2:
            height = st.number_input("🔹 키 (cm)", min_value=120, max_value=250, value=170)
            weight = st.number_input("🔹 몸무게 (kg)", min_value=30, max_value=200, value=70)
        
        st.markdown("---")
        st.markdown("### 💖 **건강 정보 입력**")
        st.write("")
        st.write("")
        col3, col4 = st.columns(2)
        with col3:
            systolic_bp = st.number_input("💓 수축기(최고) 혈압 (mmHg)", min_value=50, max_value=200, value=120)
        with col4:
            diastolic_bp = st.number_input("🩸 이완기(최저) 혈압 (mmHg)", min_value=40, max_value=150, value=80)
        
        st.write("")
        st.write("")
        st.markdown("---")
        st.markdown("### 🏃 **생활 습관 입력**")
        st.write("해당 부분에 체크해주세요 (복수 선택 가능)")
        st.write("")
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
        
        st.write("-----")
        submit = st.form_submit_button("🔮 예측하기")
        st.write("")
        st.write("")
    
    # 예측 실행 및 후처리
    if submit:
        # [1] 입력 전처리
        bp_ratio = round(systolic_bp / diastolic_bp, 2) if diastolic_bp > 0 else 0
        BMI = round(weight / ((height / 100) ** 2), 2)
        blood_pressure_diff = systolic_bp - diastolic_bp
        
        input_data = np.array([[ 
            1 if gender == "남성" else 0, 
            age, height, weight,
            smoke, alco, active, 
            systolic_bp, diastolic_bp,
            bp_ratio, BMI, blood_pressure_diff
        ]])
        
        # [2] 모델 예측 (원시 확률)
        predicted_probs = model.predict_proba(input_data)
        arr = np.array(predicted_probs)
        
        diseases = ["고혈압", "비만", "당뇨병", "고지혈증"]
        disease_probabilities = {}
        
        if arr.ndim == 3:
            if hasattr(model, "estimators_"):
                for i, disease in enumerate(diseases):
                    pos_index = list(model.estimators_[i].classes_).index(1)
                    disease_probabilities[disease] = predicted_probs[i][0][pos_index] * 100
            else:
                for i, disease in enumerate(diseases):
                    disease_probabilities[disease] = predicted_probs[i][0][1] * 100
        elif arr.ndim == 2:
            if hasattr(model, "classes_"):
                pos_index = list(model.classes_).index(1)
                for i, disease in enumerate(diseases):
                    disease_probabilities[disease] = predicted_probs[i][pos_index] * 100
            else:
                for i, disease in enumerate(diseases):
                    disease_probabilities[disease] = predicted_probs[i][1] * 100
        elif arr.ndim == 1 and len(arr) == 4:
            for i, disease in enumerate(diseases):
                disease_probabilities[disease] = predicted_probs[i] * 100
        else:
            st.error(f"예상치 못한 predict_proba() 결과 형태입니다: shape={arr.shape}")
            disease_probabilities = {d: 0 for d in diseases}
        
        # [3] '비만' 위험도 재계산 (BMI 기반)
        if BMI <= 16:
            obesity_risk = 5
        elif BMI <= 25:
            obesity_risk = ((BMI - 16) / (25 - 16)) * (50 - 5) + 5
        elif BMI <= 40:
            obesity_risk = ((BMI - 25) / (40 - 25)) * (100 - 50) + 50
        else:
            obesity_risk = 100
        disease_probabilities["비만"] = obesity_risk
        
       
        
        # [5] 라이프스타일 보정 적용
        # 고혈압: 흡연 시 +5, 음주 시 +5, 운동 시 -10  
        # 당뇨병, 고지혈증: 흡연 시 +5, (음주 효과 없음), 운동 시 -10  
        # 비만: 운동 시 -10
        for disease in disease_probabilities:
            adjusted = disease_probabilities[disease]
            if disease == "고혈압":
                if smoke:
                    adjusted += 30
                if alco:
                    adjusted -= 30
                if active:
                    adjusted += 10
            elif disease in ["당뇨병", "고지혈증",'고혈압']:
                if smoke:
                    adjusted -= 10
                if active:
                    adjusted += 10
                if alco :
                    adjusted -10

            
            disease_probabilities[disease] = min(max(adjusted, 0), 100)
        
        # [6] 나이 보정 적용 (기준 나이 50세, 70세 이상은 70세로 고정)
        effective_age = age if age <= 80 else 80
        for disease in disease_probabilities:
            if disease == "고혈압":
                adjustment = 0.5 * (effective_age - 20)
            else:
                adjustment = (effective_age - 20)
            disease_probabilities[disease] = min(max(disease_probabilities[disease] + adjustment, 0), 100)
        
        # [7] 최종 결과 출력 및 시각화
        st.markdown("---")
        st.markdown("### 📢 **건강 예측 결과**")
        st.write("")
        st.write("")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="💓 고혈압 위험", value=f"{disease_probabilities['고혈압']:.2f}%")
            st.progress(float(disease_probabilities["고혈압"]) / 100)
            st.metric(label="⚖️ 비만 위험", value=f"{disease_probabilities['비만']:.2f}%")
            st.progress(float(disease_probabilities["비만"]) / 100)
        with col2:
            st.metric(label="🍬 당뇨병 위험", value=f"{disease_probabilities['당뇨병']:.2f}%")
            st.progress(float(disease_probabilities["당뇨병"]) / 100)
            st.metric(label="🩸 고지혈증 위험", value=f"{disease_probabilities['고지혈증']:.2f}%")
            st.progress(float(disease_probabilities["고지혈증"]) / 100)
        
        st.write("")
        st.write("")
        st.write("### ✅ 건강 진단 및 조치 추천 ✅")
        def show_health_risk(disease, very_high=90, high=75, moderate=50, low=35):
            prob = disease_probabilities[disease]
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
