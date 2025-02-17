import json
import datetime
import streamlit as st
import time
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# ✅ Google Fit에서 추가 데이터를 가져올 수 있도록 SCOPES 확장
SCOPES = [
    "https://www.googleapis.com/auth/fitness.body.read",
    "https://www.googleapis.com/auth/fitness.activity.read",
    "https://www.googleapis.com/auth/fitness.blood_pressure.read",
    "https://www.googleapis.com/auth/fitness.heart_rate.read",
    "https://www.googleapis.com/auth/fitness.location.read"
]

def authenticate_google_fit():
    """Google Fit OAuth 2.0 인증"""
    flow = InstalledAppFlow.from_client_secrets_file(".streamlit/client_secret.json", SCOPES)
    creds = flow.run_local_server(port=0)
    st.query_params["authenticated"] = "true"
    return creds

def list_data_sources(creds):
    """Google Fit에서 사용 가능한 데이터 소스를 출력하여 올바른 소스를 찾음"""
    service = build("fitness", "v1", credentials=creds)
    data_sources = service.users().dataSources().list(userId="me").execute()

    source_dict = {}
    for source in data_sources.get("dataSource", []):
        print(f"📌 데이터 소스 ID: {source['dataStreamId']} - 타입: {source['dataType']['name']}")
        source_dict[source['dataType']['name']] = source['dataStreamId']

    return source_dict

def get_google_fit_data(creds):
    """Google Fit API에서 사용자 데이터 가져오기"""
    service = build("fitness", "v1", credentials=creds)
    
    # ✅ API 요청 기간을 "오늘 포함"으로 변
    now = datetime.datetime.utcnow()
    start_time = now - datetime.timedelta(days=0)  # 오늘 포함
    dataset_id = f"{int(start_time.timestamp() * 1e9)}-{int(now.timestamp() * 1e9)}"

    # ✅ 올바른 데이터 소스를 자동으로 검색하여 설정
    available_sources = list_data_sources(creds)

    data_types = {
        "height": available_sources.get("com.google.height", "raw:com.google.height:com.google.android.apps.fitness:user_input"),
        "weight": available_sources.get("com.google.weight", "raw:com.google.weight:com.google.android.apps.fitness:user_input"),
        "blood_pressure_systolic": available_sources.get("com.google.blood_pressure.systolic", "raw:com.google.blood_pressure:com.google.android.apps.fitness:user_input"),
        "blood_pressure_diastolic": available_sources.get("com.google.blood_pressure.diastolic", "raw:com.google.blood_pressure:com.google.android.apps.fitness:user_input"),
    }

    fit_data = {}

    for key, data_source in data_types.items():
        try:
            result = service.users().dataSources().datasets().get(
                userId="me", dataSourceId=data_source, datasetId=dataset_id
            ).execute()

            # ✅ API 응답 로그 출력
            print(f"🔍 {key} 데이터 요청 - 소스: {data_source}")
            print(f"🔍 {key} 데이터 응답: {json.dumps(result, indent=2, ensure_ascii=False)}")

            # ✅ 최신 데이터 추출
            if "point" in result and len(result["point"]) > 0:
                fit_data[key] = result["point"][-1]["value"][0].get("fpVal", "정보 없음")
            else:
                fit_data[key] = "정보 없음"
        except Exception as e:
            fit_data[key] = f"오류 발생: {e}"

    return fit_data

def run_ml():
    """Streamlit 앱 실행"""
    authenticated = st.query_params.get("authenticated", "false")
    st.info("🔹 로그인을 한 후 창을 닫고 돌아와 주세요.")

    if authenticated == "true":
        creds = authenticate_google_fit()
        fit_data = get_google_fit_data(creds)

        # ✅ 결과 표시
        st.success("✅ Google Fit 인증이 완료되었습니다!")
        st.write(f"📏 키: {fit_data.get('height', '정보 없음')} cm")
        st.write(f"⚖️ 몸무게: {fit_data.get('weight', '정보 없음')} kg")
        st.write(f"💓 수축기 혈압: {fit_data.get('blood_pressure_systolic', '정보 없음')} mmHg")
        st.write(f"🩸 이완기 혈압: {fit_data.get('blood_pressure_diastolic', '정보 없음')} mmHg")

        # ✅ API 응답 데이터를 Streamlit에서 JSON 형태로 표시
        st.subheader("🔍 API 응답 데이터 확인")
        st.json(fit_data)

        # ✅ 5초 후 자동으로 홈으로 이동 (JavaScript 활용)
        st.markdown(
            """
            <script>
                setTimeout(function() {
                    window.location.href = "/";
                }, 5000);
            </script>
            """,
            unsafe_allow_html=True
        )
        st.info("🔄 5초 후 자동으로 홈 화면으로 이동합니다. 창이 닫히지 않으면 수동으로 닫아주세요.")

        # ✅ "홈으로 돌아가기" 버튼
        if st.button("🏠 홈으로 돌아가기"):
            st.query_params["authenticated"] = "false"
            st.rerun()  # ✅ 최신 Streamlit 버전에서 사용
