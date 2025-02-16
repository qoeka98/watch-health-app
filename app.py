import streamlit as st
from streamlit_option_menu import option_menu
from eda import run_eda
from home import run_home
from ml import run_ml

# ✅ 전체 페이지 스타일 설정
st.set_page_config(
    page_title="건강 예측 AI",
    page_icon="🩺",
    layout="wide"
)

def main():
    # ✅ Streamlit Option Menu 사용
    with st.sidebar:
        st.markdown("<h2 style='text-align: center; color: #007bff;'>📌 건강 예측 AI</h2>", unsafe_allow_html=True)
        
        menu = option_menu(
            menu_title="",
            options=["🏠 홈", "🔍 질병 예측", "📊 개발 과정"],
            icons=["house", "stethoscope", "bar-chart-line"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#f8f9fa"},
                "icon": {"color": "blue", "font-size": "18px"}, 
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "5px",
                    "padding": "10px",
                    "border-radius": "10px",
                    "transition": "0.3s",
                    "color": "#333",
                },
                "nav-link-selected": {"background-color": "#007bff", "color": "white"},
            }
        )

    # ✅ 선택된 메뉴 실행
    if menu == "🏠 홈":
        run_home()
    elif menu == "🔍 질병 예측":
        run_eda()
    elif menu == "📊 개발 과정":
        run_ml()

# ✅ 실행
if __name__ == "__main__":
    main()
