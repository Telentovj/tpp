from distutils.command.upload import upload
import streamlit as st
import time


with open( "styles.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)



if "currentPage" not in st.session_state:
    st.session_state.currentPage = "mainPage"

st.title("Text Pre Processing")
mainPage = st.empty()
faqPage = st.empty()
insightPage = st.empty()


# Main page 
if st.session_state.currentPage == "mainPage":
    mainPage = st.container();
    with mainPage:
        col1, col2, col3 = st.columns([0.5, 0.1, 0.5])
        col2.image("csv-svgrepo-com 1csv.png", use_column_width=True)
        st.markdown("<h2 style='text-align: center;font-size: 24px;'>Preprocess your text data</h2>", unsafe_allow_html=True)

        uploaded_file = st.file_uploader("",key="enabled")

        if uploaded_file is not None:
            my_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.005)
                my_bar.progress(percent_complete + 1)
                if percent_complete == 99:
                    my_bar.empty()
                    col1, col2, col3 = st.columns([1,1,1])
                    st.markdown("""---""")
                    st.session_state.currentPage = "insightPage"
                    insight = col2.button("Click here to focus on the insights that has be found!", key="insight")
                    if insight:
                        st.session_state.currentPage = "insightPage"

                    
        format_btn = st.markdown("<h3 style='text-align: center; font-size: 14px;'>Unsure about how to format your text data?<h3>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1,1,1])

        faq = col2.button("Read our FAQs for a quick guide!", key = "faq")
        if faq:
            st.session_state.currentPage = "faqPage"
            st.write('Click again to focus on the Faqs!')
            st.markdown("""---""")







        
            


# FAQ page
if st.session_state["currentPage"] is "faqPage":
    faqPage = st.container()
    with faqPage:
        option = st.selectbox(
            'Frequently Asked Questions',
            ('How to do this?', 'How to do that?', 'How to do those?'))
        st.write(option)
        if option == 'How to do this?':
            st.write('Answer for: '+ option)
        if option == 'How to do that?':
            st.write('Answer for: '+ option)
        if option == 'How to do those?':
            st.write('Answer for: '+ option)
        
        if st.button("Close Faqs"):
            st.write("Are you sure you want to leave the Faqs?")
            st.session_state.currentPage = "mainPage"


# Insights page

if st.session_state["currentPage"] is "insightPage":
    
    insightPage = st.container()
    with insightPage:
        option = st.selectbox(
            'What do you want to know about your text file?',
            ('Select a question','What are some topics found in the data?', 'How many times did each identified topic appear?'))

        if option == 'What are some topics found in the data?':
            st.write('Answer for: '+ option)
        if option == 'How many times did each identified topic appear?':
            st.write('Answer for: '+ option)
        
        if st.button("Input another file"):
            st.write("Are you sure you want to leave the page?")
            st.session_state.currentPage = "mainPage"
            
