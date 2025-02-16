import os
import datetime
import streamlit as st
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# ✅ Google API 인증 정보
SCOPES = ["https://www.googleapis.com/auth/fitness.activity.read"]
CLIENT_SECRET_FILE = "client_secret_1077330647143-9vjkcnuvt3sdl8jta68najs64u7j1ja9.apps.googleusercontent.com.json" 

def authenticate_google_fit():
    """Google Fit API 인증 실행"""
    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
    creds = flow.run_local_server(port=0)  # 로컬 서버에서 인증 실행 (사용자 로그인 필요)
    return creds

def get_google_fit_data():
    """Google Fit에서 건강 데이터를 가져오는 함수"""
    creds = authenticate_google_fit()
    service = build("fitness", "v1", credentials=creds)

    now = datetime.datetime.utcnow()
    start_time = now - datetime.timedelta(days=1)  # 최근 하루 데이터 가져오기
    dataset_id = f"{int(start_time.timestamp() * 1e9)}-{int(now.timestamp() * 1e9)}"

    data_types = {
        "heart_rate": "derived:com.google.heart_rate.bpm:com.google.android.gms:merge_heart_rate_bpm",
        "blood_pressure": "derived:com.google.blood_pressure:com.google.android.gms:merge_blood_pressure",
        "weight": "derived:com.google.weight:com.google.android.gms:merge_weight",
        "height": "derived:com.google.height:com.google.android.gms:merge_height",
    }

    fit_data = {}

    for key, data_source in data_types.items():
        try:
            result = service.users().dataSources().datasets().get(
                userId="me", dataSourceId=data_source, datasetId=dataset_id
            ).execute()

            if "point" in result and len(result["point"]) > 0:
                fit_data[key] = result["point"][-1]["value"][0]["fpVal"]
            else:
                fit_data[key] = "정보 없음"
        except Exception:
            fit_data[key] = "정보 없음"

    return fit_data

def run_ml():
    st.title("🩺 건강 예측 AI (Google Fit 연동)")
    st.markdown("📌 **Google Fit에서 데이터를 가져와 자동으로 건강 예측을 실행합니다.**")

    # ✅ Google Fit 데이터 가져오기 버튼
    if st.button("🔄 Google Fit 데이터 가져오기"):
        google_fit_data = get_google_fit_data()
        st.session_state["google_fit_data"] = google_fit_data  # 세션 상태에 저장하여 유지

    # ✅ Google Fit 데이터가 있을 경우 자동 입력
    fit_data = st.session_state.get("google_fit_data", {})

    heart_rate = fit_data.get("heart_rate", "정보 없음")
    systolic_bp = fit_data.get("blood_pressure", "정보 없음")
    weight = fit_data.get("weight", "정보 없음")
    height = fit_data.get("height", "정보 없음")

    # ✅ 데이터 출력
    st.write(f"✅ **심박수**: {heart_rate} bpm" if heart_rate != "정보 없음" else "⚠️ **심박수 정보 없음**")
    st.write(f"✅ **수축기 혈압**: {systolic_bp} mmHg" if systolic_bp != "정보 없음" else "⚠️ **수축기 혈압 정보 없음**")
    st.write(f"✅ **몸무게**: {weight} kg" if weight != "정보 없음" else "⚠️ **몸무게 정보 없음**")
    st.write(f"✅ **키**: {height} cm" if height != "정보 없음" else "⚠️ **키 정보 없음**")

    # ✅ BMI 계산 및 예측 실행
    if "정보 없음" not in [heart_rate, systolic_bp, weight, height]:
        BMI = round(float(weight) / ((float(height) / 100) ** 2), 2)
        input_data = [[float(heart_rate), float(systolic_bp), BMI]]

        # ✅ 모델 예측 실행 (Machine Learning 모델 필요)
        # predicted_probs = model.predict_proba(input_data)
        # diseases = ["고혈압", "비만", "당뇨병", "고지혈증"]
        # disease_probabilities = {diseases[i]: predicted_probs[i][0][1] * 100 for i in range(len(diseases))}

        st.markdown("---")
        st.markdown("### 📢 **건강 예측 결과**")

        # 예측 결과 표시 (모델이 있다면 활성화)
        # for disease, prob in disease_probabilities.items():
        #     st.metric(label=f"📌 {disease} 위험", value=f"{prob:.2f}%")
        #     st.progress(prob / 100)
    else:
        st.warning("⚠️ 일부 건강 데이터가 없어서 예측을 실행할 수 없습니다.")

if __name__ == "__main__":
    run_ml()
