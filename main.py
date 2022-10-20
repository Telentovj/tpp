import streamlit as st
import time
from topic_models.data import *
from topic_models.bertopic import *
from topic_models.lda import *
from topic_models.top2vec import *
from topic_models.nmf import *

with open("styles.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

if "currentPage" not in st.session_state:
    st.session_state.currentPage = "mainPage"

st.title("Text Pre Processing")
mainPage = st.empty()
faqPage = st.empty()
insightPage = st.empty()


def change_page(page):
    st.session_state.currentPage = page


# Main page
if st.session_state.currentPage == "mainPage":
    mainPage = st.container()
    with mainPage:
        col1, col2, col3 = st.columns([0.5, 0.1, 0.5])
        col2.image("csv.png", use_column_width=True)
        st.markdown(
            "<h2 style='text-align: center;font-size: 24px;'>Preprocess your text data</h2>", unsafe_allow_html=True)

        uploaded_file = st.file_uploader("", key="enabled")

        if uploaded_file is not None:
            df = load_data(uploaded_file)
            df, docs, docs_tokenized = preprocess_data(df)
            st.session_state["dataframe"] = df
            bert = run_bertopic(docs, 3)
            samples = get_top_documents(df, bert, 3, 3)
            csv = samples_to_csv(samples)

            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='test.csv',
                mime='text/csv',
            )


            my_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.005)
                my_bar.progress(percent_complete + 1)
                if percent_complete == 99:
                    my_bar.empty()
                    col1, col2, col3 = st.columns([1, 1, 1])
                    st.markdown("""---""")
                    insight = col2.button("Click here to focus on the insights that has be found!",
                        on_click=change_page, args=("insightPage", ))
        
        format_btn = st.markdown(
            "<h3 style='text-align: center; font-size: 14px;'>Unsure about how to format your text data?<h3>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])

        faq = col2.button("Read our FAQs for a quick guide!",
                          on_click=change_page, args=("faqPage", ))


# FAQ page
if st.session_state["currentPage"] == "faqPage":
    faqPage = st.container()
    with faqPage:
        option = st.selectbox(
            'Frequently Asked Questions',
            ('How to format my excel file?', 'How to do that?', 'How to do those?'))
        st.write(option)
        if option == 'How to format my excel file?':
            st.write('Answer for: ' + option)
        if option == 'How to do that?':
            st.write('Answer for: ' + option)
        if option == 'How to do those?':
            st.write('Answer for: ' + option)

        close_faq = st.button("Close Faqs",
                          on_click=change_page, args=("mainPage", ))



# Insights page

if st.session_state["currentPage"] == "insightPage":
    insightPage = st.container()
    with insightPage:
        option = st.selectbox(
            'What do you want to know about your text file?',
            ('Select a question', 'What are some topics found in the data?', 'How many times did each identified topic appear?'))

        if option == 'What are some topics found in the data?':
            st.write('Answer for: ' + option)
        if option == 'How many times did each identified topic appear?':
            st.write('Answer for: ' + option)

        go_back = st.button("Input another file",
                on_click=change_page, args=("mainPage", ))
