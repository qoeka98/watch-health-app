import streamlit as st
from huggingface_hub import InferenceClient
import re

def clean_input(text):
    """불필요한 단어(해줘, 알려줘 등)를 제거한 사용자 입력을 반환"""
    text = re.sub(r"\b(해줘|알려줘|설명해 줘|말해 줘)\b", "", text, flags=re.IGNORECASE)
    return text.strip()

def is_health_related(text):
    """입력된 질문이 건강 관련 질문인지 판별"""
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
    """AI 응답에서 첫 줄(질문 반복) 제거"""
    response = response.replace(user_input, "").strip()
    response_lines = response.split("\n")
    if len(response_lines) > 1:
        response = "\n".join(response_lines[1:]).strip()
    return response

def get_huggingface_token():
    return st.secrets.get("HUGGINGFACE_API_TOKEN")

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

    # ✅ Llama3 모델 사용
    client = InferenceClient(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        api_key=token
    )

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "user", "content": "건강 상담을 진행해주세요"}
        ]

    for message in st.session_state.messages:
        with st.chat_message("user" if message["role"] == "user" else "assistant"):
            st.markdown(message["content"])

    chat = st.chat_input("건강 관련 질문을 입력하세요!")

    if chat:
        clean_chat = clean_input(chat)

        if not is_health_related(clean_chat):
            response = "죄송합니다. 건강 관련 상담만 가능합니다."
        else:
            # ✅ 건강 전문가 역할 프롬프트
            system_prompt = (
                "너는 전문 건강 상담 AI야. 건강, 의학, 약학, 한의학, 질병 관리, 영양, 운동 등 "
                "모든 건강 정보에 대해 전문가답게 조언해줘. 특히 비만, 당뇨, 고혈압, 고지혈증, 다이어트에 대해선 깊이 있는 상담이 가능해."
            )

            st.session_state.messages.append({"role": "user", "content": clean_chat})
            with st.chat_message("user"):
                st.markdown(clean_chat)

            with st.spinner("AI가 응답 중입니다..."):
                response = client.text_generation(
                    prompt=f"<|system|>\n{system_prompt}\n<|user|>\n{clean_chat}\n<|assistant|>",
                    max_new_tokens=512,
                    temperature=0.7,
                )

            response = filter_ai_response(response, clean_chat)

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
