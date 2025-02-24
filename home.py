import streamlit as st
import os

# ✅ 페이지 설정

def run_home():
    # ✅ 페이지 제목
    st.markdown("<h1 style='text-align: center; color: #007bff;'>🏠 건강 예측 AI 홈</h1>", unsafe_allow_html=True)
    st.info("💡 **고혈압, 비만, 당뇨병, 고지혈증 4개의 질병을 예측해 우리 함께 건강한 삶을 만들어 갑시다!** 🏥")

    # ✅ 이미지 삽입 (경로 확인)
    image_path = "image/진료.png"
    if os.path.exists(image_path):
        st.image(image_path, width=500)
    else:
        st.warning("⚠️ 이미지를 찾을 수 없습니다. 올바른 경로인지 확인해주세요.")

    st.write("")
    st.write("")
    st.write("")
    st.write("")
    # ✅ 소개 섹션 (카드 스타일 적용)
    st.markdown(
        """
        <div style="
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            border-left: 5px solid #007bff;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);">
            <h3>🤖 AI 기반 건강 예측 시스템</h3>
            <p>이 애플리케이션은 AI 모델을 활용하여 <b>고혈압, 비만, 당뇨, 고지혈증</b>의 위험도를 분석합니다.</p>
            <p>📌 <b>질병 예측</b> 메뉴에서 자신의 건강 데이터를 입력하면 AI가 위험도를 예측해드립니다!</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")
    st.write("")
    st.write("")
    st.write("")

   

if __name__ == "__main__":
    run_home()
