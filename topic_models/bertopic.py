import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer


def run_bertopic(docs, num_topics):
    """
    Runs BERTopic on provided documents (docs) and outputs topics (num_topics)

    Args:
    docs -> List of documents
    num_topics -> int

    Returns:
    - Trained "model" that can be used to return visualizations and stats
    """
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

    vectorizer_model = CountVectorizer(stop_words="english")

    model = BERTopic(
        nr_topics=num_topics,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
    )

    topics, probabilities = model.fit_transform(docs)

    return model

def get_top_documents_bert(df, model, k):
    """
    Args:
    - df: pandas dataframe
    - model: bertopic model
    - k: how many sample to be extracted per topic

    Returns:
    - samples: Array of sample documents
    - topic_numbers: Array of corresponding topic numbers
    - topic_words: String of representative topic_words for topic
    """
    samples = []
    topic_numbers = []
    topic_words = []
    representative_docs = model.representative_docs_
    for topic_num, documents in representative_docs.items():
        topic_word = " ".join(list(map(lambda x: x[0], model.get_topic(topic_num))))
        for index, doc in enumerate(documents):
            if index > k:
                break
            sample = df.loc[df["processed"] == doc]["text"].values[0]
            samples.append(sample)
            topic_numbers.append(topic_num)
            topic_words.append(topic_word)
    return samples, topic_numbers, topic_words

def samples_to_csv_bert(samples, topic_numbers, topic_words):
    """
    Converts list of samples, topic_numbers and topic_words
    to output an encoded csv for streamlit

    Args:
    - samples: Array of sample documents
    - topic_numbers: Array of corresponding topic numbers
    - topic_words: String of representative topic_words for topic 

    Returns:
    - CSV object
    """
    data = {
        'text': samples,
        'topic_number': topic_numbers,
        'topic_word': topic_words
    }
    df = pd.DataFrame(data)
    return df.to_csv().encode('utf-8')
