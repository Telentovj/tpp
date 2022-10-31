import streamlit as st
import math
from topic_models.data import *
from topic_models.bertopic import *
from topic_models.lda import *
from topic_models.top2vec import *
from topic_models.nmf import *
import streamlit.components.v1 as components

st.set_page_config(layout="wide")

with open("styles.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

if "currentPage" not in st.session_state:
    st.session_state.currentPage = "main_page"

st.title("Text Pre Processing")
main_page = st.empty()
faq_page = st.empty()
insight_page = st.empty()
download_page = st.empty()

# For changing pages
def change_page(page):
    st.session_state.currentPage = page


# For input widget within insight page to get number of sample per topic
def set_topic_model(model):
    num_topics = 0
    if not st.session_state["auto"]:
        num_topics = st.session_state["number_of_topics"]
    else: 
        if model == "top2vec":
            num_topics = st.session_state["top2vec_number_of_topics"]
        #include bert check
    total_sample_size = num_topics * st.session_state["k"]
    size_of_data_set = len(st.session_state["docs"])
    if total_sample_size < size_of_data_set:
        if st.session_state["k"]:
            st.session_state.topicModel = model
            change_page("download_page")
        else:
            st.warning("Set number of topics.")
    else:
        st.warning(
            "Please set a lower number of samples per topic. The max you can set is: "
            + str(size_of_data_set)
        )


# For checkbox widget to toggle usage of model
def set_model_usage(session_state_name, current_session_state_value):
    if current_session_state_value:
        st.session_state[session_state_name] = False
    else:
        st.session_state[session_state_name] = True



# Main page
if st.session_state.currentPage == "main_page":
    main_page = st.container()
    with main_page:
        col1, col2, col3 = st.columns([0.5, 0.1, 0.5])
        col2.image("csv.png", use_column_width=True)
        st.markdown(
            "<h2 style='text-align: center;font-size: 24px;'>Preprocess your text data</h2>",
            unsafe_allow_html=True,
        )

        # Faqs
        col1, col2, col3 = st.columns([1, 0.25, 1])
        faq = col2.button("Read our FAQs!", on_click=change_page, args=("faq_page",))

        # Initalise session states for model usage, defaults to True.
        if "use_bert" not in st.session_state:
            st.session_state["use_bert"] = True
        if "use_lda" not in st.session_state:
            st.session_state["use_lda"] = True
        if "use_top2vec" not in st.session_state:
            st.session_state["use_top2vec"] = True
        if "use_nmf" not in st.session_state:
            st.session_state["use_nmf"] = True
        if "auto" not in st.session_state:
            st.session_state["auto"] = False

        # Checkboxes for selecting which models to use
        auto_generate = st.checkbox(
            "Automatically generate topics", 
            key="disable",
            on_change=set_model_usage,
            args=(
                "auto",
                st.session_state["auto"],
            ),
        )


        

        use_bert, use_lda, use_top2vec, use_nmf = st.columns([1, 1, 1, 1])
        use_bert.checkbox(
            "Use Bert Model",
            value=st.session_state["use_bert"],
            on_change=set_model_usage,
            args=(
                "use_bert",
                st.session_state["use_bert"],
            ),
        )
        use_lda.checkbox(
            "Use LDA Model",
            value=st.session_state["use_lda"],
            on_change=set_model_usage,
            disabled=st.session_state.disable,
            args=(
                "use_lda",
                st.session_state["use_lda"],
            ),
        )
        use_top2vec.checkbox(
            "Use Top2Vec Model",
            value=st.session_state["use_top2vec"],
            on_change=set_model_usage,
            args=(
                "use_top2vec",
                st.session_state["use_top2vec"],
            ),
        )
        use_nmf.checkbox(
            "Use NMF Model",
            value=st.session_state["use_nmf"],
            on_change=set_model_usage,
            disabled=st.session_state.disable,
            args=(
                "use_nmf",
                st.session_state["use_nmf"],
            ),
        )

        
        
        # Input for number of topics
        number_of_topics = st.number_input(
            "Insert number of Topics, decimals will be rounded down.",
            min_value=1,
            max_value=10,
            value=3,
            disabled=st.session_state.disable
        )

        #if auto_generate:



        # File uploader
        uploaded_file = st.file_uploader("", type=["csv", "xlsx"], key="enabled")

        # add logic to ensure that number of topics is not None
        if uploaded_file is not None:
            if not auto_generate:

                if number_of_topics:

                    # Column for in progress text
                    col1, col2, col3, col4 = st.columns([0.25, 0.25, 0.25, 0.25])
                    if st.session_state["use_bert"]:
                        col1.write("Awaiting Bert Process to Begin")
                    if st.session_state["use_lda"]:
                        col2.write("Awaiting LDA  Process to Begin")
                    if st.session_state["use_top2vec"]:
                        col3.write("Awaiting Top2Vec Process to Begin")
                    if st.session_state["use_nmf"]:
                        col4.write("Awaiting NMF Process to Begin")

                    number_of_topics = math.floor(number_of_topics)
                    st.session_state["number_of_topics"] = number_of_topics

                    df = load_data(uploaded_file, uploaded_file.name)
                    df, docs, docs_tokenized = preprocess_data(df)
                    st.session_state["dataframe"] = df
                    st.session_state["docs"] = docs
                    st.session_state["docs_tokenized"] = docs_tokenized

                    # Bert logic
                    if st.session_state["use_bert"]:
                        col1.write("Running Bert.....")
                        bert = run_bertopic(docs, number_of_topics)
                        st.session_state["bert"] = bert
                        col1.write("Bert Model Completed")

                    # Lda logic
                    if st.session_state['use_lda']:
                        col2.write("Running LDA.....")
                        lda_model, bow_corpus, dictionary = run_lda(docs_tokenized, number_of_topics)
                        st.session_state["lda"] = lda_model
                        st.session_state["bow_corpus"] = bow_corpus
                        st.session_state["lda_dictionary"] = dictionary
                        col2.write("LDA Model Completed")

                    # top2vec logic
                    if st.session_state['use_top2vec']:
                        col3.write("Running Top2Vec.....")
                        top2vec = runTop2Vec(docs)
                        st.session_state["top2vec"] = top2vec
                        runTop2VecReduced(top2vec, number_of_topics)
                        col3.write("Top2Vec Model Completed")

                    # nmf logic
                    if st.session_state["use_nmf"]:
                        col4.write("Running NMF.....")
                        nmf, tfidf_feature_names = run_nmf(docs, number_of_topics)
                        st.session_state["nmf"] = nmf
                        st.session_state["tfidf_feature_names"] = tfidf_feature_names
                        st.session_state["running_nmf"] = False
                        col4.write("NMF Model Completed")

                    insight1, insight2, insight3 = st.columns([1, 0.5, 1])
                    insight = insight2.button(
                        "Click here to focus on the insights that has be found!",
                        on_click=change_page,
                        args=("insight_page",),
                    )
            elif auto_generate:
                col1, col2 = st.columns([0.5, 0.5])
                if st.session_state["use_bert"]:
                    col1.write("Awaiting Bert Process to Begin")

                if st.session_state["use_top2vec"]:
                    col2.write("Awaiting Top2Vec Process to Begin")
                df = load_data(uploaded_file, uploaded_file.name)
                df, docs, docs_tokenized = preprocess_data(df)
                st.session_state["dataframe"] = df
                st.session_state["docs"] = docs
                st.session_state["docs_tokenized"] = docs_tokenized

                if st.session_state['use_top2vec']:
                    col2.write("Running Top2Vec.....")
                    top2vec = runTop2Vec(docs)
                    st.session_state["top2vec"] = top2vec
                    col2.write("Top2Vec Model Completed")

                #insert bert auto topic finder

                insight1, insight2, insight3 = st.columns([1, 0.5, 1])
                insight = insight2.button(
                    "Click here to focus on the insights that has be found!",
                    on_click=change_page,
                    args=("insight_page",),
                )

            else:
                st.warning("Please insert the number of topics.")


# FAQ page
if st.session_state["currentPage"] == "faq_page":
    faq_page = st.container()
    with faq_page:
        option = st.selectbox(
            "Frequently Asked Questions",
            ("How to format my excel file?", "How to do that?", "How to do those?"),
        )
        st.write("Answer for: " + option)
        close_faq = st.button("Close Faqs", on_click=change_page, args=("main_page",))


# Insights page
if st.session_state["currentPage"] == "insight_page":
    insight_page = st.container()
    if not st.session_state["auto"]:
        number_of_topics = st.session_state["number_of_topics"]
        st.session_state["top2vec_number_of_topics"] = number_of_topics
        #include bert num of topics
    else:
        st.session_state["top2vec_number_of_topics"] = st.session_state['top2vec'].get_num_topics()
        #include bert num of topics

    with insight_page:

        # WordCloud
        word_cloud_expander = st.expander("Word cloud for entire dataset")
        word_cloud_expander.pyplot(wordcloud(st.session_state["docs_tokenized"]))

        # BERT
        if st.session_state["use_bert"]:
            bert = st.session_state["bert"]
            bert_expander = st.expander("Bert")
            bert_expander.write(
                bert.visualize_barchart().update_layout(
                    autosize=False, width=670, height=400
                )
            )

        # Top2Vec
        if st.session_state['use_top2vec']:
            top2vec = st.session_state['top2vec']
            top2vec_expander = st.expander("Top2Vec")
            number_of_topics = st.session_state["top2vec_number_of_topics"]
            if not st.session_state["auto"]:
                for i in range(number_of_topics):
                    fig = printWordBarReduced(top2vec, i)
                    top2vec_expander.plotly_chart(fig, use_container_width=True)
            else:
                for i in range(number_of_topics):
                    fig = printWordBar(top2vec, i)
                    top2vec_expander.plotly_chart(fig, use_container_width=True)

        # LDA
        if not st.session_state["auto"]:
            if st.session_state['use_lda']:
                lda = st.session_state['lda']
                with st.expander("LDA"):
                    col1, col2, col3 = st.columns([0.5, 0.1, 0.5])
                    components.html(visualize_chart_lda(
                        lda,
                        st.session_state['bow_corpus'],
                        st.session_state['lda_dictionary']
                    ), width=1300, height=800, scrolling=True)

        # NMF
        
            if st.session_state["use_nmf"]:
                nmf = st.session_state["nmf"]
                tfidf_feature_names = st.session_state["tfidf_feature_names"]
                NMF_expander = st.expander("NMF")
                NMF_expander.pyplot(
                    plot_top_words(
                        nmf,
                        tfidf_feature_names,
                        10,
                        "Topics in NMF model (KL Divergence Loss)",
                    )
                )

        # Ask for how many datapoints you want her topic, k.
        k = st.number_input(
            "Insert number of datapoints, you want for each topic, decimals will be rounded down.",
            min_value=1,
            max_value=100,
            value=5,
        )
        st.session_state["k"] = k

        # Generate buttons to go to download page
        col1, col2, col3, col4 = st.columns([0.25, 0.25, 0.25, 0.25])
        generate_with_bert = col1.button(
            "Generate dataset with Bert",
            on_click=set_topic_model,
            args=("bert",),
            disabled=(st.session_state["use_bert"] == False),
        )
        generate_with_top2vec = col2.button(
            "Generate dataset with Top2Vec",
            on_click=set_topic_model,
            args=("top2vec",),
            disabled=(st.session_state['use_top2vec'] == False),
        )
        generate_with_lda = col3.button(
                                "Generate dataset with LDA",
                                on_click=set_topic_model,
                                args=("lda",),
                                disabled=(st.session_state['use_lda'] == False),
                            )
        generate_with_nmf = col4.button(
            "Generate dataset with NMF",
            on_click=set_topic_model,
            args=("nmf",),
            disabled=(st.session_state["use_nmf"] == False),
        )

        go_back = st.button(
            "Go Back to Main Page", on_click=change_page, args=("main_page",)
        )

# Download Page
if st.session_state["currentPage"] == "download_page":
    download_page = st.container()
    topic_model = st.session_state["topicModel"]
    number_of_topics = 0
    if not st.session_state["auto"]:
        number_of_topics = st.session_state["number_of_topics"]


    bow_corpus = st.session_state["bow_corpus"]
    docs = st.session_state["docs"]
    df = st.session_state["dataframe"]
    k = st.session_state["k"]

    if topic_model == "bert" and st.session_state["use_bert"]:
        bert = st.session_state["bert"]
        df = get_top_docs_bert(df, bert, k)
        labeled_csv = df_to_csv(df)

    if topic_model == "top2vec" and st.session_state['use_top2vec']:
        top2vec = st.session_state['top2vec']
        if not st.session_state["auto"]:
            df = get_top_documents_Top2Vec_reduced(df, top2vec, number_of_topics, k)
        else:
            df = get_top_documents_Top2Vec(df, top2vec, st.session_state["top2vec_number_of_topics"], k)
        labeled_csv = df_to_csv(df)

    if topic_model == "lda" and st.session_state['use_lda']:
        lda = st.session_state['lda']
        samples, topic_numbers, topic_words, topic_scores = get_top_documents_lda(df, bow_corpus, lda, number_of_topics, k)
        labeled_csv = samples_to_csv(samples, topic_numbers, topic_words, topic_scores)

    if topic_model == "nmf" and st.session_state["use_nmf"]:
        nmf = st.session_state["nmf"]
        samples = get_top_docs_nmf(df, docs, nmf, number_of_topics, k)
        labeled_csv = samples_to_csv(samples)

    with download_page:
        st.write("Download dataset labeled with: " + topic_model)
        st.download_button(
            label="Download data as CSV",
            data=labeled_csv,
            file_name="test.csv",
            mime="text/csv",
        )

        go_back = st.button(
            "Go Back to Insights", on_click=change_page, args=("insight_page",)
        )

        go_back_to_start = st.button(
            "Input another file", on_click=change_page, args=("main_page",)
        )
