import streamlit as st

with open( "styles.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

if "title" not in st.session_state:
    st.session_state["title"] = "Text Pre Processing"

print(st.session_state["title"])
test = st.title(st.session_state["title"])

if st.button('test'):

    if st.session_state["title"] == "Text Pre Processing":
        st.session_state["title"] = "b"
    
    elif st.session_state["title"] == "b":
        st.session_state["title"] = "Text Pre Processing"