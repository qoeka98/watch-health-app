import streamlit as st
from huggingface_hub import InferenceClient
import re

def clean_input(text):
    """불필요한 단어 제거"""
    text = re.sub(r"\b(해줘|알려줘|설명해 줘|말해 줘)\b", "", text, flags=re.IGNORECASE)
    return text.strip()

def is_health_related(text):
    """건강 관련 키워드 탐색"""
    health_keywords = [
        "건강", "의학", "의료", "약학", "한의학", "당뇨", "비만", "고지혈증", "고혈압",
        "운동", "영양", "콜레스테롤", "혈압", "혈당", "체중", "심장", "신장", "식습관",
        "혈액 검사", "당뇨병", "저혈압", "체질량", "콜레스테롤 수치", "암", "폐암", "간암",
        "위암", "대장암", "심장병", "뇌졸중", "심근경색", "협심증", "치매", "우울증",
        "스트레스", "천식", "간경화", "신부전", "위염", "장염", "소화불량", "갑상선",
        "류마티스", "관절염", "식이요법", "영양소", "칼슘", "철분", "단백질", "비타민",
        "섭취량", "칼로리", "저염식", "고단백", "채식", "비건", "키토제닉", "간헐적 단식",
        "면역력", "스트레스 관리", "수면", "불면증", "건강검진", "예방접종", "운동법",
        "근력 운동", "유산소 운동", "요가", "필라테스", "명상", "호흡법", "BMI", 
        "심전도", "혈액형", "빈혈 검사", "건강상태", "질병"
    ]
    return any(keyword in text for keyword in health_keywords)

def filter_ai_response(response, user_input):
    """AI 응답에서 반복되는 질문 제거"""
    response = response.replace(user_input, "").strip()
    response_lines = response.split("\n")
    if len(response_lines) > 1:
        response = "\n".join(response_lines[1:]).strip()
    return response

def get_huggingface_token():
    # 🔁 로컬 개발 시 직접 입력 / 배포 시 secrets.toml 사용
    return "hf_your_token_here"  # 여기에 본인의 HF API 토큰 입력

def run_snagdam():
    st.title("💬 건강 상담 챗봇")

    st.info(
        '''건강 예측을 바탕으로 건강 상담을 진행해보세요! 🩺  

        **예시 질문:**  
        - 건강 상태를 개선하려면 어떤 운동이 좋을까요?  
        - 식단을 어떻게 조절하면 좋을까요?  
        - 특정 질병의 위험을 낮추기 위한 생활 습관은?  
        - 정기 건강 검진은 얼마나 자주 받아야 하나요?  
        - 혈압을 낮추는 방법에는 무엇이 있나요?  
        '''
    )

    token = get_huggingface_token()

    # ✅ Mistral 모델로 설정
    client = InferenceClient(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        token=token
    )

    # ✅ 초기 메시지 세팅
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "user", "content": "건강 상담을 시작해주세요"}
        ]

    for message in st.session_state.messages:
        with st.chat_message("user" if message["role"] == "user" else "assistant"):
            st.markdown(message["content"])

    # ✅ 사용자 입력 받기
    chat = st.chat_input("건강 관련 질문을 입력하세요!")

    if chat:
        clean_chat = clean_input(chat)

        if not is_health_related(clean_chat):
            response = "죄송합니다. 건강 관련 질문만 답변할 수 있어요."
        else:
            # ✅ 시스템 프롬프트 (역할 지정)
            system_prompt = (
                "너는 건강 전문가야. 당뇨, 고혈압, 고지혈증, 비만, 다이어트 등 "
                "의학적 정보를 신중하고 정확하게 설명해줘야 해."
            )

            # ✅ 사용자 메시지 저장 및 UI 출력
            st.session_state.messages.append({"role": "user", "content": clean_chat})
            with st.chat_message("user"):
                st.markdown(clean_chat)

            # ✅ AI 응답
            with st.spinner("AI가 응답을 생성 중입니다..."):
                full_prompt = f"[INST] {system_prompt}\n\n{clean_chat} [/INST]"

                response = client.text_generation(
                    prompt=full_prompt,
                    max_new_tokens=512,
                    temperature=0.7
                )

                response = filter_ai_response(response, clean_chat)

        # ✅ 응답 저장 및 출력
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

# ✅ 메인 함수 진입점
def main():
    run_snagdam()


