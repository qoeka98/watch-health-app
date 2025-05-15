import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

# ✅ 1. 모델 로드 (최초 1회만)
@st.cache_resource
def load_model():
    model_name = "beomi/KoAlpaca-Polyglot-12.8B"  # 또는 'skt/kogpt2-base-v2' 등
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# ✅ 사용자 입력 정제
def clean_input(text):
    text = re.sub(r"\b(해줘|알려줘|설명해 줘|말해 줘)\b", "", text, flags=re.IGNORECASE)
    return text.strip()

# ✅ 건강 키워드 필터
def is_health_related(text):
    return any(k in text for k in ["건강", "운동", "질병", "영양", "식이요법", "혈압", "비만", "당뇨", "수면"])

# ✅ Streamlit UI
st.title("🧠 건강 상담 챗봇 (KoLLM 로컬 실행)")
st.info("한국어 LLM을 로컬에서 직접 실행한 건강 상담 챗봇입니다.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# ✅ 이전 메시지 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("건강 관련 질문을 입력해주세요!")

if user_input:
    clean_chat = clean_input(user_input)
    st.session_state.messages.append({"role": "user", "content": clean_chat})
    with st.chat_message("user"):
        st.markdown(clean_chat)

    if not is_health_related(clean_chat):
        ai_response = "❗죄송합니다. 건강 관련 질문만 상담할 수 있습니다."
    else:
        # 프롬프트 구성
        system_prompt = (
            "당신은 건강 전문 상담 AI입니다. "
            "운동, 식단, 질병 예방에 대해 정확하고 공손하게 300자 이내로 설명하세요.\n\n"
            f"질문: {clean_chat}\n\n답변:"
        )

        # 토크나이징 및 생성
        input_ids = tokenizer.encode(system_prompt, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_length=300,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        ai_response = tokenizer.decode(output[0], skip_special_tokens=True)
        ai_response = ai_response.replace(system_prompt, "").strip()

    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    with st.chat_message("assistant"):
        st.markdown(ai_response)
