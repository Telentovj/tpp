import gensim
import pyLDAvis.gensim_models
import pandas as pd


def run_lda(docs_tokenized, num_topics):

    """
    Runs LDA on provided documents (docs) and number of topics (num_topics)

    Args:
    docs -> List of tokens
    num_topics -> int

    Returns:
    - lda_model:Trained "model" that can be used to return visualizations and stats
    - bow_corpus: dataset in bag of words form
    - dictionary: dictionary of tokens and their id
    """
    dictionary = gensim.corpora.Dictionary(docs_tokenized)
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    bow_corpus = [dictionary.doc2bow(doc) for doc in docs_tokenized]
    lda_model = gensim.models.LdaModel(
        bow_corpus, num_topics=num_topics, id2word=dictionary, passes=2
    )

    return (lda_model, bow_corpus, dictionary)


def get_all_docs_lda(df, bow_corpus, model):
    """
    Args:
    - df: pandas dataframe with columns: text
    - bow_corpus: Bag of Words
    - model: LDA model

    Returns:
    - Dataframe with columns 'doc', 'topic_label'. This is all docs from the dataset (docs)
    """

    topic_label = [
        max(model.get_document_topics(bow), key=lambda tup: tup[1])[0]
        for bow in bow_corpus
    ]

    new_df = pd.DataFrame({"doc": df.text, "topic_label": topic_label})

    return new_df


def get_top_documents_lda(df, bow_corpus, model, num_topics, k):

    """
    Args:
    - df: pandas dataframe with columns: text
    - bow_corpus: vectorised tokens
    - model: LDA model
    - num_topics: number of topics generated by model
    - k: how many sample to be extracted per topic

    Returns:
    - samples: list of sample documents
    - topic_numbers: list of dominant topic number for each document
    - topic_words: list of topic words for the topic
    - topic_scores: topic score for the document for this topic
    """

    samples = []
    topic_numbers = []
    topic_words = []
    topic_scores = []

    df[["topic_label", "topic_score"]] = [
        max(model.get_document_topics(bow), key=lambda tup: tup[1])
        for bow in bow_corpus
    ]

    for topic_num in range(num_topics):
        df_selected_topic = df[df["topic_label"] == topic_num].copy()
        added_samples = list(df_selected_topic["text"][:k].values)
        samples = samples + added_samples
        words = " ".join([x[0] for x in model.show_topic(topic_num, topn=10)])
        topic_words = topic_words + [words] * len(added_samples)
        topic_numbers = topic_numbers + [topic_num] * len(added_samples)
        topic_scores = topic_scores + list(df_selected_topic["topic_score"][:k].values)

    # Convert to a proper dataframe
    data = {"doc": samples, "topic_label": topic_numbers, "topic_words": topic_words}
    return pd.DataFrame(data)


def get_topic_lda(model, topic_num):
    """
    Returns top n words for a specific topic and their c-tf-idf scores
    - list of Tuples (word, score)
    """
    return model.show_topic(topic_num)


def get_topic_freq_lda(df, bow_corpus, model):

    """
    Return a Series object whereby the value is the size of the topics (in descending order) with the index as the topic number
    """
    df["dominant_topic"] = [
        max(model.get_document_topics(bow), key=lambda tup: tup[1])[0]
        for bow in bow_corpus
    ]
    topic_freq = df.dominant_topic.value_counts().sort_values(ascending=False)

    return topic_freq


def visualize_chart_lda(model, bow_corpus, dictionary):
    """
    Returns a Figure object that I assume plots out the barchart,
    Doesnt plot on CLI
    """
    vis = pyLDAvis.gensim_models.prepare(model, bow_corpus, dictionary)
    return pyLDAvis.prepared_data_to_html(vis)
