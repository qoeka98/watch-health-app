import streamlit as st
import os

def run_home():
    st.title('당신의 질병을 예측해드립니다!')

    image_path = "image/진료.png"
    
    if os.path.exists(image_path):
        st.image(image_path, use_column_width=True)
    else:
        st.warning("이미지를 찾을 수 없습니다. 올바른 경로인지 확인해주세요.")
