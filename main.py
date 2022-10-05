import streamlit as st

with open( "styles.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)



if "currentPage" not in st.session_state:
    st.session_state.currentPage = "mainPage"


mainPage = st.empty()
faqPage = st.empty()
InsightPage = st.empty()

# Main page 
if st.session_state.currentPage == "mainPage":
    mainPage = st.container();
    with mainPage:
        st.title("Text Pre Processing")
        col1, col2, col3 = st.columns([0.5, 0.1, 0.5])
        col2.image("csv-svgrepo-com 1csv.png", use_column_width=True)
        st.markdown("<h2 style='text-align: center;font-size: 24px;'>Preprocess your text data</h2>", unsafe_allow_html=True)
        st.file_uploader("")
        st.markdown("<h3 style='text-align: center; font-size: 14px;'>Unsure about how to format your text data?</h3>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1,1,1])
        if col2.button("Read our FAQs for a quick guide!"):
            st.session_state.currentPage = "faqPage"
            


# FAQ page
if st.session_state["currentPage"] is "faqPage":
    with st.container():
        col1, col2, col3 = st.columns([0.5, 0.1, 0.5])
        col2.image("csv-svgrepo-com 1csv.png", use_column_width=True)


# Loading page

# Insights drop

#