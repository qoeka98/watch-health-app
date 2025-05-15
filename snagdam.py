import streamlit as st
from huggingface_hub import InferenceClient
import re

# ✅ 사용자 입력 정제
def clean_input(text):
    text = re.sub(r"\b(해줘|알려줘|설명해 줘|말해 줘)\b", "", text, flags=re.IGNORECASE)
    return text.strip()

# ✅ 건강 관련 키워드 판별
def is_health_related(text):
    health_keywords = [
        "건강", "의학", "의료", "약학", "한의학", "당뇨", "비만", "고지혈증", "고혈압",
        "운동", "영양", "콜레스테롤", "혈압", "혈당", "체중", "심장", "신장", "식습관",
        "혈액 검사", "당뇨병", "저혈압", "체질량", "콜레스테롤 수치",
        "암", "위암", "간암", "대장암", "심장병", "뇌졸중", "심근경색", "협심증", 
        "치매", "파킨슨병", "우울증", "불안장애", "스트레스", "알츠하이머", "천식",
        "간경화", "신부전", "위염", "장염", "소화불량", "갑상선", "류마티스", "관절염",
        "다이어트", "식이요법", "영양소", "칼슘", "철분", "단백질", "비타민", "미네랄",
        "섭취량", "칼로리", "저염식", "고단백", "채식", "비건", "간헐적 단식",
        "면역력", "수면", "건강검진", "예방접종", "운동법", "건강개선", "식단",
        "유산소 운동", "요가", "필라테스", "명상", "호흡법",
        "혈당 수치", "혈압 수치", "체지방률", "BMI", "콜레스테롤 검사", "간 수치",
        "신장 기능 검사", "심전도", "빈혈 검사", "질병"
    ]
    return any(keyword in text for keyword in health_keywords)

# ✅ 응답에서 질문 제거 및 첫 줄 제거
def filter_ai_response(response, user_input):
    response = response.replace(user_input, "").strip()
    response_lines = response.split("\n")
    if len(response_lines) > 1:
        response = "\n".join(response_lines[1:]).strip()
    return response

# ✅ Hugging Face API 토큰
def get_huggingface_token():
    return st.secrets.get("HUGGINGFACE_API_TOKEN")

# ✅ 챗봇 실행
def run_snagdam():
    st.title("💬 건강 상담 챗봇")
    st.info(
        '''건강 예측을 바탕으로 건강 상담을 진행해보세요! 🩺

        **예시 질문:**  
        - 건강 상태를 개선하려면 어떤 운동이 좋을까요?  
        - 식단을 어떻게 조절하면 좋을까요?  
        - 특정 질병의 위험을 낮추기 위한 생활 습관은?  
        - 혈압을 낮추는 방법에는 무엇이 있나요?  
        '''
    )

    token = get_huggingface_token()
    client = InferenceClient(
        model="HuggingFaceH4/zephyr-7b-beta",
        api_key=token
    )

    # ✅ 초기 세션 메시지
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ✅ 이전 메시지 출력
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ✅ 사용자 입력
    chat = st.chat_input("건강 관련 질문을 입력하세요!")

    if chat:
        clean_chat = clean_input(chat)

        if not is_health_related(clean_chat):
            response = "죄송합니다. 건강 관련 질문만 상담할 수 있습니다."
        else:
            # ✅ system prompt 개선
            system_prompt = (
                "당신은 건강 전문 상담 AI입니다. "
                "질병 예방, 운동, 식이요법 등 건강과 관련된 질문에 정확하고 친절하게 답변해주세요. "
                "너무 반복적인 표현은 피하고, 신뢰할 수 있는 정보를 바탕으로 설명해주세요."
                " 사용자가 질문한 내용을 바탕으로 답변을 작성해주세요. "
                "질문에 대한 답변은 간결하고 명확하게 300자넘지 않게 작성해주세요. "
                "문장은 반드시 완결되도록 끝맺어주세요."
                "문장은 반드시 공손히 문장을 완성해주세요."
            )

            full_prompt = system_prompt + "\n\n질문: " + clean_chat

            # ✅ 사용자 메시지 저장
            st.session_state.messages.append({"role": "user", "content": clean_chat})
            with st.chat_message("user"):
                st.markdown(clean_chat)

            with st.spinner("AI가 응답을 생성 중입니다..."):
                try:
                    raw_response = client.text_generation(
                        prompt=full_prompt,
                        max_new_tokens=300
                    )
                    response = filter_ai_response(raw_response, clean_chat)

                    # ✅ 반복적인 이상 응답 필터링
                    if any(keyword in response for keyword in ["스쿨지어", "스탭스타이저", "가슴과 허벅지", "같은 운동"]):
                        response = "죄송합니다. 질문을 조금 더 구체적으로 입력해 주세요. 다시 안내해드릴게요."

                except Exception as e:
                    response = f"⚠️ 오류가 발생했습니다: {e}"

        # ✅ AI 응답 저장 및 출력
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
