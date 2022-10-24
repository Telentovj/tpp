import streamlit as st
import math
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
downloadPage = st.empty()

def change_page(page):
    st.session_state.currentPage = page

def set_topic_model(model):
    st.session_state.topicModel = model
    change_page("downloadPage")


# Main page
if st.session_state.currentPage == "mainPage":
    mainPage = st.container()
    with mainPage:
        col1, col2, col3 = st.columns([0.5, 0.1, 0.5])
        col2.image("csv.png", use_column_width=True)
        st.markdown(
            "<h2 style='text-align: center;font-size: 24px;'>Preprocess your text data</h2>", unsafe_allow_html=True)
        
        # Faqs
        col1, col2, col3 = st.columns([1, 0.55, 1])
        faq = col2.button("Read our FAQs!",
                          on_click=change_page, args=("faqPage", ))
        
        # Input for number of topics 
        number_of_topics = st.number_input(
            'Insert number of Topics, decimals will be rounded down.',
            min_value = 0, 
            max_value= 10,
        )

        # File uploader
        uploaded_file = st.file_uploader("", key="enabled")

        # add logic to ensure that number of topics is not None
        if uploaded_file is not None:
            if number_of_topics:
                number_of_topics = math.floor(number_of_topics)
                df = load_data(uploaded_file)
                df, docs, docs_tokenized = preprocess_data(df)
                st.session_state["dataframe"] = df
                
                # Bert logic
                bert = run_bertopic(docs, 4)
                st.session_state["bert"] = bert
                # samples = get_top_documents(df, bert, 3, 3)
                # csv = samples_to_csv(samples)

                # st.download_button(
                #     label="Download data as CSV",
                #     data=csv,
                #     file_name='test.csv',
                #     mime='text/csv',
                # )

                # Lda logic
                lda = run_lda(docs,4)

                # nmf logic
                # nmf = run_nmf(docs,4)

                # top2vec logic
                # top2vec = runTop2Vec(docs)

                my_bar = st.progress(0)
                for percent_complete in range(100):
                    time.sleep(0.005)
                    my_bar.progress(percent_complete + 1)
                    if percent_complete == 99:
                        my_bar.empty()
                        st.markdown("""---""")
                        col1, col2, col3 = st.columns([1, 1, 1])
                        insight = col2.button("Click here to focus on the insights that has be found!",
                                            on_click=change_page, args=("insightPage", ))
            else:
                st.warning('Please insert the number of topics.')
            




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
        
        #BERT
        bert = st.session_state['bert']
        bert_expander = st.expander("Bert")
        bert_expander.write(bert.visualize_barchart().update_layout(autosize=False,width = 670,height=400))

        #Top2Vec
        Top2Vec_expander = st.expander("Top2Vec")


        #LDA 
        LDA_expander = st.expander("LDA")


        #NMF
        NML_expander = st.expander("NMF")


        col1, col2, col3, col4 = st.columns([0.25,0.25,0.25,0.25])
        generate_with_a = col1.button("Generate dataset with Bert",
                            on_click=set_topic_model, args=("bert", ))
        generate_with_b = col2.button("Generate dataset with Top2Vec",
                            on_click=set_topic_model, args=("top2vec", ))
        generate_with_c = col3.button("Generate dataset with LDA",
                            on_click=set_topic_model, args=("lda", ))
        generate_with_c = col4.button("Generate dataset with NMF",
                            on_click=set_topic_model, args=("nmf", ))

# Download Page

if st.session_state["currentPage"] == "downloadPage":
    downloadPage = st.container()
    with downloadPage:
        st.write(st.session_state["topicModel"])
        go_back = st.button("Input another file",
                            on_click=change_page, args=("mainPage", ))
