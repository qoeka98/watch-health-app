import streamlit as st
import numpy as np
import joblib

def main():
    # ✅ 모델 불러오기 (저장된 모델)
    model = joblib.load("classifier2_model.pkl")

    # ✅ 웹 앱 제목
    st.title("🩺 건강 예측 AI")

    st.write("🔍 아래 정보를 입력하면 AI가 건강 위험도를 예측합니다.")

    # ✅ 사용자 입력을 받기 위한 UI 요소
    gender = st.radio("성별", ["여성", "남성"])
    height = st.number_input("키 (cm)", min_value=120, max_value=250, value=170)
    weight = st.number_input("몸무게 (kg)", min_value=30, max_value=200, value=70)
    systolic_bp = st.number_input("수축기(최고) 혈압 (mmHg)", min_value=50, max_value=200, value=120)
    diastolic_bp = st.number_input("이완기(최저) 혈압 (mmHg)", min_value=40, max_value=150, value=80)
    smoke = st.radio("흡연 여부", ["비흡연", "흡연"])
    alco = st.radio("음주 여부", ["비음주", "음주"])
    active = st.radio("운동 여부", ["운동 안함", "운동 함"])

    # ✅ 혈압 차이 자동 계산
    blood_pressure_diff = systolic_bp - diastolic_bp

    # ✅ 입력 데이터 변환
    input_data = np.array([[
        1 if gender == "남성" else 0,  # 성별 변환 (남성=1, 여성=0)
        height, weight, systolic_bp, diastolic_bp,
        1 if smoke == "흡연" else 0, 
        1 if alco == "음주" else 0,   
        1 if active == "운동 함" else 0, 
        blood_pressure_diff
    ]])

    # ✅ 예측 버튼 클릭 시 실행
    if st.button("🔮 예측하기"):
        # AI 모델 예측 (확률 기반)
        predicted_probs = model.predict_proba(input_data)

        # ✅ 질병 이름 목록
        diseases = ["고혈압", "비만", "당뇨병", "고지혈증"]

        # ✅ 확률을 100%로 변환
        disease_probabilities = {disease: predicted_probs[i][0][1] * 100 for i, disease in enumerate(diseases)}

        # ✅ 예측 결과 출력
        st.write("### 📢 건강 예측 결과:")
        for disease, probability in disease_probabilities.items():
            st.write(f"✅ **{disease} 확률:** {probability:.2f}%")

        # ✅ 최종 건강 진단 & 조치 추천
        st.write("\n### ✅ 최종 건강 진단 & 조치 추천 ✅")

        # **고혈압 (hypertension)**
        if disease_probabilities["고혈압"] > 90:
            st.error("🚨 **고혈압 위험이 매우 높습니다! 즉각적인 혈압 관리가 필요합니다. 병원 방문을 추천합니다.**")
        elif disease_probabilities["고혈압"] > 70:
            st.warning("⚠️ **고혈압 위험이 높습니다. 꾸준한 혈압 측정과 저염식이 필요합니다.**")
        else:
            st.success("✅ **고혈압 위험이 낮습니다. 건강한 식습관을 유지하세요.**")

        # **비만 (obesity)**
        if disease_probabilities["비만"] > 90:
            st.error("🚨 **고도비만 상태입니다. 체중 감량을 위한 생활습관 개선이 필요합니다.**")
        elif disease_probabilities["비만"] > 70:
            st.warning("⚠️ **과체중 상태입니다. 식단 관리와 운동을 병행하세요.**")
        else:
            st.success("✅ **비만 위험이 낮습니다. 건강한 체중을 유지하세요.**")

        # **당뇨병 (diabetes)**
        if disease_probabilities["당뇨병"] > 50:
            st.warning("⚠️ **당뇨병 위험이 높습니다. 혈당을 주기적으로 체크하고, 당 섭취를 줄이세요.**")
        elif disease_probabilities["당뇨병"] > 20:
            st.info("🟡 **당뇨병 위험이 보통입니다. 건강한 식단을 유지하세요.**")
        else:
            st.success("✅ **당뇨병 위험이 낮습니다.**")

        # **고지혈증 (hyperlipidemia)**
        if disease_probabilities["고지혈증"] > 50:
            st.warning("⚠️ **고지혈증(고콜레스테롤) 위험이 높습니다. 지방 섭취를 줄이고 운동을 병행하세요.**")
        elif disease_probabilities["고지혈증"] > 20:
            st.info("🟡 **고지혈증 위험이 보통입니다. 정기적인 건강 검진을 받으세요.**")
        else:
            st.success("✅ **고지혈증 위험이 낮습니다.**")

        st.write("\n🩺 **건강을 위해 정기적인 검진을 받고, 생활습관을 개선하세요!** 🚀")

if __name__ == "__main__":
    main()
