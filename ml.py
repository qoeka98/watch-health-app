import streamlit as st

def run_ml():
    st.markdown("""
    ## 📌 **건강 예측 AI & 상담 챗봇 개발 과정** 🚀  
    ---

    # 🏥 **프로젝트 개요**  
    이 프로젝트는 **Streamlit 기반 웹 애플리케이션**으로,  
    ✅ **건강 위험 예측 AI** 🤖  
    ✅ **AI 기반 건강 상담 챗봇** 💬  
    두 가지 기능을 통합하여 **사용자가 자신의 건강 상태를 예측하고, 필요한 건강 상담을 받을 수 있도록** 설계되었습니다.

    ---
                
     ## 🏗 **입력 데이터(X) 가공하여 성능 최적화**  
    🤔 **기존 데이터만으로 충분할까? NO!**  

        기존 컬럼
                 
        age	나이
        gender	성별
        height	키 (cm)
        weight	몸무게 (kg)
        ap_hi	수축기 혈압 (최고 혈압, mmHg)
        ap_lo	이완기 혈압 (최저 혈압, mmHg)
        cholesterol	총 콜레스테롤 수치
        gluc	혈당 수치
        smoke	흡연 여부 (0: 비흡연, 1: 흡연)
        alco	음주 여부 (0: 비음주, 1: 음주)
        active	신체 활동 여부 (0: 비활동적, 1: 활동적)

    💡 **추가 변수 도입!**  
    📌 질병과 직접적인 관계가 있는 변수들 추가
        BMI= BMI 계산
        hypertension고혈압
        obesity비만 여부
        diabetes당뇨병 여부
        hyperlipidemia 고지혈증 여부


    📌 체질량 및 혈압 관련 심층 변수 도입
        bp_ratio 혈압 비율
        bmi_category비만 여부
        cholesterol_high 고콜레스테롤 여부
        blood_pressure_diff 혈압 차이
        bmi_ap_hi_ratioBMI와 수축기 혈압 비율
        bmi_ap_lo_ratioBMI와 이완기 혈압 비율

    📌 비만과 질병의 관계를 고려한 변수 추가

        obesity_hyperlipidemia 비만 & 고지혈증 동시 보유
        obesity_diabetes비만 & 당뇨병 동시 보유

    🚬 흡연 & 🍺 음주가 질병 위험을 어떻게 증가시키는지 반영

        smoke_hypertension흡연 & 고혈압
        smoke_hyperlipidemia 흡연 & 고지혈증
        smoke_diabetes 흡연 & 당뇨병

        alco_hypertension음주 & 고혈압
        alco_hyperlipidemia 음주 & 고지혈증
        alco_diabetes음주 & 당뇨병

    🚬🍺 흡연 & 음주가 동시에 영향을 미치는 경우도 반영
        smoke_alco_hypertension

    📌 운동 여부 및 연령을 고려한 변수 추가
        age 나이를 연도로 변환(일수로되어있었습니다)
        age_group 연령대 구분

    📌 혈압과 질병의 관계를 반영한 변수 추가

        bp_high_risk매우 높은 혈압 위험군
        bp_diabetes_risk고혈압 + 혈당 이상 조합

 

    📊 **이러한 변수를 추가한 결과,**  
    ✔️ **정확도 87.1%까지 향상**!  
    ✔️ **질병 예측의 신뢰도 증가**!  

    ---

    ## 🔬 **질병 예측 AI 모델 개발 과정**  

    건강 위험도를 예측하기 위해 **랜덤 포레스트(Random Forest) 분류기**와 **XGBoost 분류기**를 비교하며 **최적의 모델을 탐색**했습니다.  

    ### 🔥 **RandomForestClassifier vs XGBoostClassifier**  
    💡 **어떤 모델이 더 뛰어날까?**  
    ✔️ RandomForest는 단순하면서도 직관적인 모델!  
    ✔️ XGBoost는 강력한 부스팅 기법을 활용하여 성능을 극대화!  

    |  모델  | 정확도 (Accuracy) | 정밀도 (Precision) | 재현율 (Recall) | AUC-ROC  |
    |:----:|:------------:|:------------:|:------------:|:--------:|
    | **RandomForest** | 81.4% | 79.6% | 77.2% | 84.1%  |
    | **XGBoost** | **85.7%** | **83.2%** | **80.5%** | **88.6%** |

    🎯 **결론:**  
    ✅ **XGBoost가 전반적인 성능이 뛰어나 최종 채택!**  
    ✅ **AUC-ROC 88.6% 달성, 더 높은 신뢰도의 질병 예측 가능!**  

    ---

    ## 🏆 **Multi-Output Classification 적용**  
    📊 **고혈압, 비만, 당뇨, 고지혈증을 동시에 예측**하기 위해,  
    각 질병을 **개별적인 클래스로 분류하고 확률(%)로 변환하여 직관적으로 제공**하도록 설계했습니다!  

    **➡️ 질병별 위험도를 0~100% 범위로 표시하여 가독성을 극대화!**  

    ---

    ## ⚠️ **데이터 불균형 문제 해결**  

    👀 **질병 데이터의 심각한 불균형!**  
    📉 일부 질병(예: 당뇨병, 고지혈증)의 데이터 수가 부족하여 모델 학습이 불균형하게 진행됨  

    🚀 **해결책?**  
    1️⃣ **SMOTE (Synthetic Minority Over-sampling Technique)**  
    2️⃣ **ADASYN (Adaptive Synthetic Sampling)**  
                
     **[ADASYN사용] 으로 불균형 해결!**

    ---

   

    # 💬 **AI 건강 상담 챗봇 개발**  

    ### 🎯 **LLM 기반 챗봇 설계**  
    ✅ **Hugging Face의 `google/gemma-2-9b-it` 모델 채택!**  
    ✅ **의료 상담 특화 프롬프트 엔지니어링 적용!**  

    ### 🔥 **프롬프트 최적화 & 모델 튜닝**  
    📌 시스템 프롬포트를 활용한 프롬푸트 최적화!
    📌 깔끔한 응답 생성과 건강지식에 특화된 모델로 진화!  

    🎯 **결과:**  
    ✔️ **더 자연스럽고 의료적인 답변 제공 가능!**  
    ✔️ **의료 상담 관련 질문이 아닐 경우, 필터링하여 응답 조정!**  

    ---

    # 🔐 **API 관리 및 보안 강화**  

    ✅ **Hugging Face API 활용**  
    ✅ **API 키 보안 강화를 위해 `.gitignore`에 API 키 추가**  
    ✅ **배포 시 API 키 노출 방지하여 안전한 운영 가능!**  

    ---

    # 🚀 **Streamlit을 활용한 웹 애플리케이션 개발**  

    ### 🏗 **앱 UI 구성**  
    📌 **Streamlit 기반의 직관적인 UI**  
    📌 **사이드바에서 쉽게 페이지 이동 가능! (`streamlit_option_menu` 활용)**  

    ### 🔥 **질병 예측 & 건강 상담 챗봇 통합**  
    ✅ **질병 예측 결과를 직관적인 `st.metric`을 활용해 % 단위로 표시**  
    ✅ **Plotly 바 그래프를 통해 대한민국 평균과 비교 가능!**  
    ✅ **예측 결과 기반 건강 상담 챗봇 연동!**  

    ---

    # 🚀 **배포 및 유지보수**  
    📌 **배포를 위한 `requirements.txt` 작성**  
    📌 **Hugging Face API를 사용하여 원활한 모델 실행**  
    📌 **.gitignore에 API 키 포함하여 보안 강화**  

    ---

    # 🎯 **결론**  
    이 프로젝트를 통해,  

    ✅ **AUC-ROC 90.3%의 신뢰할 수 있는 질병 예측 AI 구축!**  
    ✅ **Hugging Face `google/gemma-2-9b-it` 모델을 활용한 건강 상담 챗봇 개발!**  
    ✅ **Streamlit 기반으로 간편하게 배포할 수 있는 웹 애플리케이션 완성!**  

    🔥 **AI 기반 건강 예측 & 상담 서비스의 새로운 가능성을 제시한 프로젝트!** 🚀
    """)
    st.write("")

    st.write("-------")

    st.write("")

    st.markdown('''데이터 출쳐  : https://www.kaggle.com/datasets/akshatshaw7/cardiovascular-disease-dataset'''
                )
    st.markdown('''개발자 깃허브 : https://github.com/qoeka98/watch-health-app'''
                )
    st.markdown('''개발자 이메일 : vhzkflfltm6@gmail.com''')

if __name__ == "__main__":
    run_ml()
