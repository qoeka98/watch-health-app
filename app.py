import streamlit as st


from eda import run_eda
from home import run_home
from ml import run_ml


from ml import run_ml
def main():
    menu = st.sidebar.radio("메뉴 선택", ["🏠 홈", "🔍 질병예측", "📊 개발과정"])

    if menu == "🏠 홈":
        run_home()

    elif menu == "🔍 질병예측":
        run_eda()


    elif menu =="📊 개발과정":
        
        run_ml()

if __name__ == "__main__":
    main()
