import streamlit as st
from transformers import pipeline

# ✅ 로컬에서 바로 실행되는 text-generation 파이프라인 (Mistral 기반)
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1")

generator = load_model()

def run_snagdam():
    st.title("💬 건강 상담 챗봇 (로컬 실행)")

    st.info("이 모델은 토큰 없이 로컬에서 실행됩니다. 첫 로딩에는 시간이 걸릴 수 있습니다.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("건강 관련 질문을 입력하세요!")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("AI가 응답 중입니다..."):
            prompt = f"[INST] 건강 전문가로서 답해줘: {user_input} [/INST]"
            output = generator(prompt, max_new_tokens=256, do_sample=True)[0]['generated_text']

        response = output.replace(prompt, "").strip()
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

def main():
    run_snagdam()
