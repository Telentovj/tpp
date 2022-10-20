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

    model = BERTopic(nr_topics=num_topics, vectorizer_model=vectorizer_model, ctfidf_model=ctfidf_model)

    topics, probabilities = model.fit_transform(docs)

    return model

def get_topic_bert(model, topic_num):
    """
        Returns top n words for a specific topic and their c-tf-idf scores
        - Array of Tuples (word, score)
    """
    return model.get_topic(topic_num)

def get_topic_freq(model):
    """
        Return the size of the topics (in descending order)
    """
    return model.get_topic_freq()

def visualize_barchart_bert(model):
    """
        Returns a Figure object that I assume plots out the barchart, 
        Doesnt plot on CLI 
    """
    return model.visualize_barchart()
