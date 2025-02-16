import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def run_eda():
    # ✅ 한글 폰트 설정 (맑은 고딕 적용)
    plt.rc("font", family="Malgun Gothic")  # Windows 환경
    plt.rcParams["axes.unicode_minus"] = False  # 마이너스(-) 부호 깨짐 방지

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

    # ✅ 대한민국 평균값 (남녀 구분)
    avg_values_female = {
        "키 (cm)": 157.49,
        "몸무게 (kg)": 57.86,
        "수축기 혈압 (mmHg)": 114.82,
        "이완기 혈압 (mmHg)": 72.38,
        "혈압 차이 (mmHg)": 42.44
    }

    avg_values_male = {
        "키 (cm)": 171.22,
        "몸무게 (kg)": 71.87,
        "수축기 혈압 (mmHg)": 120.16,
        "이완기 혈압 (mmHg)": 77.89,
        "혈압 차이 (mmHg)": 42.27
    }

    # ✅ 사용자 입력값
    user_values = {
        "키 (cm)": height,
        "몸무게 (kg)": weight,
        "수축기 혈압 (mmHg)": systolic_bp,
        "이완기 혈압 (mmHg)": diastolic_bp,
        "혈압 차이 (mmHg)": blood_pressure_diff
    }

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

        # ✅ 평균값과 사용자의 입력값 비교 차트 생성
        st.write("\n### 📊 평균 vs. 입력값 비교")
        st.info('''유저가 입력한 값과 국민건강영양조사에 기반한 결과 일반적인 건강지표(실제 성인의 평균과 다소 차이가 있을 수 있습니다)를 비교한 표입니다.
                자신이 얼마나 건강한지 혹은 건강을 챙겨야하는지 평균과 비교하여 자신의 건강상태를 한눈에 알아 봅시다! ''')

        # ✅ 성별에 맞는 평균값 선택
        avg_values = avg_values_male if gender == "남성" else avg_values_female

        # ✅ Matplotlib 그래프 생성
        fig, ax = plt.subplots()

        # ✅ X축 라벨
        labels = list(avg_values.keys())

        # ✅ 평균 데이터 (각 지표의 평균값)
        avg_data = list(avg_values.values())

        # ✅ 사용자 입력 데이터
        user_data = list(user_values.values())  

        # ✅ X축 위치
        x = np.arange(len(labels))
        width = 0.35  # 바 차트 너비

        # ✅ 바 차트 생성
        ax.bar(x - width/2, avg_data, width, label="대한민국 건강 평균값", color="blue", alpha=0.6)
        ax.bar(x + width/2, user_data, width, label="유저의 입력값", color="red", alpha=0.7)

        # ✅ 그래프 설정 (한글 폰트 적용됨)
        ax.set_xlabel("건강 지표")
        ax.set_ylabel("수치")
        ax.set_title("평균값과 입력값 비교")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.legend()

        # ✅ Streamlit에 그래프 표시
        st.pyplot(fig)

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

        st.write("\n🩺 **건강을 위해 정기적인 검진을 받고, 생활습관을 개선하세요!** 🚀")

