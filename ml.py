import datetime
import os
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow

# ✅ Google Fit API 권한 (OAuth 및 서비스 계정)
SCOPES = ["https://www.googleapis.com/auth/fitness.activity.read"]
SERVICE_ACCOUNT_FILE = "service_account.json"  # 서비스 계정 JSON 파일
CLIENT_SECRET_FILE = "client_secret.json"  # OAuth 2.0 클라이언트 파일


def authenticate_google_fit():
    """Google Fit API OAuth 2.0 인증 (Streamlit 환경에서도 동작하도록 수정)"""

    if not os.path.exists(CLIENT_SECRET_FILE):
        st.error("⚠️ OAuth 인증 파일 (`client_secret.json`)이 없습니다. Google Cloud에서 다운로드하여 추가하세요.")
        return None

    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)

    # ✅ Streamlit에서는 로컬 서버 방식이 안 될 수 있으므로 직접 로그인 URL 제공
    auth_url, _ = flow.authorization_url(prompt="consent")

    st.info("Google Fit 데이터를 가져오려면 아래 링크를 클릭하여 로그인하세요.")
    st.markdown(f"[🔗 Google 로그인하기]({auth_url})", unsafe_allow_html=True)

    # ✅ 사용자가 입력할 수 있도록 UI 제공
    auth_code = st.text_input("인증 코드를 입력하고 Enter를 눌러주세요:")

    if auth_code:
        creds = flow.fetch_token(code=auth_code)
        return creds
    else:
        return None  # 인증이 완료되지 않으면 None 반환


def get_user_google_fit_data():
    """Google Fit에서 건강 데이터를 가져오는 함수"""

    creds = None

    # ✅ 서비스 계정이 존재하면 우선 사용 (Google Fit 관리자 계정 필요)
    if os.path.exists(SERVICE_ACCOUNT_FILE):
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES
        )
    else:
        # ✅ 일반 사용자는 OAuth 2.0 인증을 통해 로그인
        creds = authenticate_google_fit()

    if not creds:
        return None  # 인증 실패 시 종료

    # ✅ Google Fit API 클라이언트 빌드
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
        except KeyError:
            fit_data[key] = "정보 없음"
        except Exception as e:
            fit_data[key] = f"오류 발생: {e}"

    return fit_data


def run_ml():
    st.title("🩺 건강 예측 AI (Google Fit 연동)")
    st.markdown("📌 **Google Fit 데이터를 가져와 자동으로 건강 예측을 실행합니다.**")

    # ✅ Google Fit 데이터 가져오기 버튼
    if st.button("🔄 Google Fit 데이터 가져오기"):
        google_fit_data = get_user_google_fit_data()
        if google_fit_data:
            st.session_state["google_fit_data"] = google_fit_data  # ✅ 세션 상태에 저장
        else:
            st.error("⚠️ Google Fit 데이터를 가져오는 데 실패했습니다. 인증 문제일 수 있습니다.")

    # ✅ Google Fit 데이터가 있을 경우 자동 입력
    fit_data = st.session_state.get("google_fit_data", {})

    heart_rate = fit_data.get("heart_rate", "정보 없음")
    systolic_bp = fit_data.get("blood_pressure", "정보 없음")
    weight = fit_data.get("weight", "정보 없음")
    height = fit_data.get("height", "정보 없음")

    # ✅ 데이터 출력
    st.subheader("📊 Google Fit 건강 데이터")
    st.write(f"✅ **심박수**: {heart_rate} bpm" if heart_rate != "정보 없음" else "⚠️ **심박수 정보 없음**")
    st.write(f"✅ **수축기 혈압**: {systolic_bp} mmHg" if systolic_bp != "정보 없음" else "⚠️ **수축기 혈압 정보 없음**")
    st.write(f"✅ **몸무게**: {weight} kg" if weight != "정보 없음" else "⚠️ **몸무게 정보 없음**")
    st.write(f"✅ **키**: {height} cm" if height != "정보 없음" else "⚠️ **키 정보 없음**")


if __name__ == "__main__":
    run_ml()
