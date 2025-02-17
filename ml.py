import json
import datetime
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
def run_ml():
    # Google Fit API 권한 설정 (키, 몸무게, 혈압 가져오기)
    SCOPES = ["https://www.googleapis.com/auth/fitness.body.read"]

    def authenticate_google_fit():
        """Google Fit API 인증 실행"""
        flow = InstalledAppFlow.from_client_secrets_file("client_secret.json", SCOPES)
        creds = flow.run_local_server(port=0)
        return creds

    def get_google_fit_data(creds):
        """Google Fit API에서 사용자 데이터 가져오기"""
        service = build("fitness", "v1", credentials=creds)
        
        # 데이터셋 ID (최근 7일)
        now = datetime.datetime.utcnow()
        start_time = now - datetime.timedelta(days=7)
        dataset_id = f"{int(start_time.timestamp() * 1e9)}-{int(now.timestamp() * 1e9)}"

        # Google Fit 데이터 유형 설정 (키, 몸무게, 혈압)
        data_types = {
            "height": "derived:com.google.height:com.google.android.gms:merge_height",
            "weight": "derived:com.google.weight:com.google.android.gms:merge_weight",
            "blood_pressure_systolic": "derived:com.google.blood_pressure.systolic:com.google.android.gms:merge_blood_pressure",
            "blood_pressure_diastolic": "derived:com.google.blood_pressure.diastolic:com.google.android.gms:merge_blood_pressure",
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
            except Exception as e:
                fit_data[key] = f"오류 발생: {e}"

        return fit_data

    # 실행
    creds = authenticate_google_fit()
    fit_data = get_google_fit_data(creds)

    # 결과 출력
    print(f"키: {fit_data.get('height', '정보 없음')} cm")
    print(f"몸무게: {fit_data.get('weight', '정보 없음')} kg")
    print(f"수축기 혈압: {fit_data.get('blood_pressure_systolic', '정보 없음')} mmHg")
    print(f"이완기 혈압: {fit_data.get('blood_pressure_diastolic', '정보 없음')} mmHg")
