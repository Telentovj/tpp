from cProfile import label
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
    st.session_state.currentPage = "main_page"

st.title("Text Pre Processing")
main_page = st.empty()
faq_page = st.empty()
insight_page = st.empty()
download_page = st.empty()

def change_page(page):
    st.session_state.currentPage = page

def set_topic_model(model):
    if st.session_state['k']:
        st.session_state.topicModel = model
        change_page("download_page")
    else:
        st.warning("Set number of topics.")


# Main page
if st.session_state.currentPage == "main_page":
    main_page = st.container()
    with main_page:
        col1, col2, col3 = st.columns([0.5, 0.1, 0.5])
        col2.image("csv.png", use_column_width=True)
        st.markdown(
            "<h2 style='text-align: center;font-size: 24px;'>Preprocess your text data</h2>", unsafe_allow_html=True)
        
        # Faqs
        col1, col2, col3 = st.columns([1, 0.55, 1])
        faq = col2.button("Read our FAQs!",
                          on_click=change_page, args=("faq_page", ))
        
        # Input for number of topics 
        number_of_topics = st.number_input(
            'Insert number of Topics, decimals will be rounded down.',
            min_value = 1, 
            max_value= 10,
            value = 3,
        )

        # File uploader
        uploaded_file = st.file_uploader("", key="enabled")

        # add logic to ensure that number of topics is not None
        if uploaded_file is not None:
            if number_of_topics:
                number_of_topics = math.floor(number_of_topics)
                st.session_state['number_of_topics'] = number_of_topics

                df = load_data(uploaded_file)
                df, docs, docs_tokenized = preprocess_data(df)
                st.session_state["dataframe"] = df
                st.session_state["docs"] = docs
                
                # Bert logic
                bert = run_bertopic(docs, 4)
                st.session_state["bert"] = bert

                # Lda logic
                # wang fei this is broken :/
                # lda_model, bow_corpus, dictionary = run_lda(docs_tokenized,4)
                # st.session_state["lda"] = lda
                # st.session_state["bow_corpus"] = bow_corpus
                # st.session_state["lda_dictionary"] = dictionary

                # nmf logic
                nmf,tfidf_feature_names = run_nmf(docs,4)
                st.session_state["nmf"] = nmf
                st.session_state["tfidf_feature_names"] = tfidf_feature_names

                # top2vec logic
                # jegan this is broken :/
                # top2vec = runTop2Vec(docs)
                # st.session_state["top2vec"] = top2vec


                insight = col2.button(
                        "Click here to focus on the insights that has be found!",
                        on_click=change_page, 
                        args=("insight_page",)
                    )
            else:
                st.warning('Please insert the number of topics.')
            

# FAQ page
if st.session_state["currentPage"] == "faq_page":
    faq_page = st.container()
    with faq_page:
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
                              on_click=change_page, args=("main_page", ))


# Insights page
if st.session_state["currentPage"] == "insight_page":
    insight_page = st.container()
    number_of_topics = st.session_state['number_of_topics']

    with insight_page:
        
        #BERT
        bert = st.session_state['bert']
        bert_expander = st.expander("Bert")
        bert_expander.write(bert.visualize_barchart().update_layout(autosize=False,width = 670,height=400))

        # #Top2Vec
        # john this is broken sync with jegan
        # top2vec = st.session_state['top2vec']
        # Top2Vec_expander = st.expander("Top2Vec")

        # #LDA
        # john this is broken sync with wang fei 
        # lda = st.session_state['lda']
        # LDA_expander = st.expander("LDA")

        # NMF
        nmf = st.session_state['nmf']
        tfidf_feature_names = st.session_state['tfidf_feature_names']
        NMF_expander = st.expander("NMF")
        NMF_expander.write(
            plot_top_words(
                nmf,
                tfidf_feature_names,
                10, 
                "Topics in NMF model (KL Divergence Loss)"
            )
        )

        # Ask for how many datapoints you want her topic, k.
        k = st.number_input(
            'Insert number of datapoints, you want for each topic, decimals will be rounded down.',
            min_value = 1, 
            max_value= 100,
            value = 5,
        )

        st.session_state['k'] = k

        col1, col2, col3, col4 = st.columns([0.25, 0.25, 0.25, 0.25])
        generate_with_bert = col1.button("Generate dataset with Bert",
                            on_click=set_topic_model, args=("bert", ))
        generate_with_top2vec = col2.button("Generate dataset with Top2Vec",
                            on_click=set_topic_model, args=("top2vec", ))
        generate_with_lda = col3.button("Generate dataset with LDA",
                            on_click=set_topic_model, args=("lda", ))
        generate_with_nmf = col4.button("Generate dataset with NMF",
                            on_click=set_topic_model, args=("nmf", ))


# Download Page
if st.session_state["currentPage"] == "download_page":
    download_page = st.container()
    topic_model = st.session_state["topicModel"]
    number_of_topics = st.session_state["number_of_topics"]
    # bow_corpus = st.session_state["bow_corpus"]
    docs = st.session_state['docs']
    df = st.session_state["dataframe"]
    k  = st.session_state['k'] 

    if topic_model == "bert":
        bert = st.session_state['bert']
        samples = get_top_documents_bert(df, bert, number_of_topics, k)
        labeled_csv = samples_to_csv(samples)

    if topic_model == "top2vec":
        top2vec = st.session_state['top2vec']
        samples = get_top_documents_Top2Vec(df, top2vec, number_of_topics, k)
        labeled_csv = samples_to_csv(samples)

    if topic_model == "lda":
        lda = st.session_state['lda']
        samples = get_top_documents_lda(df, bow_corpus, lda, number_of_topics, k)
        labeled_csv = samples_to_csv(samples)

    if topic_model == "nmf":
        nmf = st.session_state['nmf']
        samples = get_top_docs_nmf(df, nmf, number_of_topics, k)
        labeled_csv = samples_to_csv

    with download_page:
        st.write("Download dataset labeled with: " + topic_model)
        st.download_button(
            label = "Download data as CSV",
            data = labeled_csv,
            file_name = 'test.csv',
            mime = 'text/csv',
        )
        go_back = st.button("Input another file",
                            on_click=change_page, args=("main_page", ))
