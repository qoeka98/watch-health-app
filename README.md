# watch-health-app

# 🏥 AI 기반 건강 예측 & 상담 챗봇 🚀

🔗 **배포 링크**: [🚀https://watch-health-app-ljurrnkvhldnnkpdsvzrrh.streamlit.app/)  


![healthcare_ai_banner](https://user-images.githubusercontent.com/your-image-url.png)

**Streamlit 기반 웹 애플리케이션**  
✅ **AI 기반 질병 예측** 🤖  
✅ **AI 건강 상담 챗봇** 💬  
✅ **Hugging Face LLM(`google/gemma-2-9b-it`) 활용**  
✅ **XGBoost + ADASYN으로 정보 불균형 해결**  

---

## 📌 프로젝트 개요

이 프로젝트는 **Streamlit을 활용한 AI 건강 예측 웹 애플리케이션**입니다.  
**XGBoost 모델을 활용한 질병 예측 시스템**과 **LLM 기반 건강 상담 챗봇**을 통합하여,  
사용자가 자신의 건강 위험도를 예측하고 필요한 상담을 받을 수 있도록 설계되었습니다.

✅ **질병 예측**: **고혈압, 비만, 당뇨병, 고지혈증** 위험도를 예측  
✅ **상담 챗봇**: AI가 건강 관련 질문을 분석하고 답변 제공  


---

## 🔬 질병 예측 모델 개발 과정

건강 위험도를 예측하기 위해 **RandomForestClassifier**와 **XGBoostClassifier**를 비교하여 최적의 모델을 선정했습니다.

### 🔥 **RandomForestClassifier vs XGBoostClassifier 비교**
| 모델  | 정확도 (Accuracy) | 정밀도 (Precision) | 재현율 (Recall) | AUC-ROC |
|:----:|:------------:|:------------:|:------------:|:--------:|
| **RandomForest** | 81.4% | 79.6% | 77.2% | 84.1%  |
| **XGBoost** | **85.7%** | **83.2%** | **80.5%** | **88.6%** |

✅ **결론**: **XGBoost가 성능이 뛰어나 최종 선택!**  
✅ **AUC-ROC 88.6% 달성 → 신뢰할 수 있는 질병 예측 가능!**  

---

## 🏆 Multi-Output Classification 적용

📌 **질병(고혈압, 비만, 당뇨, 고지혈증)을 개별 클래스로 다뤄 예측 확률을 %로 변환**  
📌 **출력 결과를 `st.metric()`을 활용해 직관적으로 제공**  

---

## ⚠️ 데이터 불균형 해결

⚠️ **질병 데이터의 심각한 불균형** → 일부 질병(당뇨병, 고지혈증)의 데이터 수 부족  
✅ **해결책**: **ADASYN(Adaptive Synthetic Sampling) 적용**  
📌 SMOTE와 비교 후 **ADASYN이 더 적합**하여 최종 선택  
📌 **AUC-ROC 90.3%까지 성능 향상!**  

---

## 🏗 X 데이터 가공하여 성능 최적화

✔️ **BMI (체질량지수)** = `weight / (height^2)`  
✔️ **혈압 비율(bp_ratio)** = `systolic_bp / diastolic_bp`  
✔️ **혈압 차이(blood_pressure_diff)** = `systolic_bp - diastolic_bp`  

✅ **추가 변수 도입 후 정확도 87.1%까지 향상!**  

---

## 💬 LLM 기반 상담 챗봇 개발

✅ **Hugging Face의 `google/gemma-2-9b-it` 모델 채택**  
✅ **프롬프트 엔지니어링 적용하여 더 자연스러운 답변 생성**  
✅ **"알려줘", "해줘" 등 불필요한 문구 제거로 응답 최적화**  

✔️ **Hugging Face API 활용하여 챗봇 실행**  
✔️ **API 키 보안 강화를 위해 `.gitignore`에 API 키 포함**  

---

## 🔐 API 보안 및 배포 과정

✅ **배포를 위해 `requirements.txt` 파일 작성**  
✅ **Hugging Face API 활용하여 모델 실행**  
✅ **API 키는 `.gitignore`에 추가하여 보안 강화**  

---

## 🚀 Streamlit 기반 UI 구성

📌 **Streamlit을 활용하여 직관적인 UI 설계**  
📌 **Plotly 그래프를 사용하여 건강 예측 결과 시각화**  
📌 **예측 결과를 기반으로 AI 상담 챗봇과 연동**  

---
## 🎯 결론 및 기대 효과

✅ **AUC-ROC 90.3%의 신뢰할 수 있는 질병 예측 AI 구축**
✅ **google/gemma-2-9b-it LLM 기반 건강 상담 챗봇 개발**
✅ **Streamlit으로 쉽고 빠르게 배포 가능한 웹 애플리케이션 완성**

🚀 AI 기반 건강 예측 & 상담 서비스의 새로운 가능성을 제시한 프로젝트!



