################################################################
#                         LOADING DATA                         #
################################################################
import sys
import pandas as pd

file_path = sys.argv[1]

df = pd.read_csv(file_path)

################################################################
#                      DATA PREPROCESSING                      #
################################################################
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer 
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from wordcloud import WordCloud
import matplotlib.pyplot as plt


df["text"] = df["text"].str.lower()
df["text"] = df["text"].replace("na", None)
text = df[~df["text"].isna()][['text']]

# Remove @, # and links
text["text"] = text["text"].str.replace(r"@[A-Za-z0-9_]+",'', regex=True)
text["text"] = text["text"].str.replace(r"#[A-Za-z0-9_]+",'', regex=True)
text["text"] = text["text"].str.replace(r"http[s]?://\S+",'', regex=True)

added_stopwords = ["http", "bgaabha", "bgaabhdit"] # to be changed based on context

def lemmatize(text):
  return WordNetLemmatizer().lemmatize(text, pos='v')

def preprocess(text):
  result = ''
  text=str(text)
  token_words = gensim.utils.simple_preprocess(text)
  for token in token_words:
    if token not in gensim.parsing.preprocessing.STOPWORDS and token not in added_stopwords and len(token) > 3:
      result = result + ' ' + lemmatize(token)
  return result

# Example
doc_sample = text[text.index == 2].values[0][0]
print('original document: ')
words = []
for word in doc_sample.split(' '):
  words.append(word)

print(doc_sample)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(doc_sample))

processed_docs = text["text"].map(preprocess)

docs = list(processed_docs)


################################################################
#                             NMF                              #
################################################################

################################################################
#                             LDA                              #
################################################################


################################################################
#                           TOP2VEC                            #
################################################################
from top2vec import Top2Vec

# docs is a array of strings, numTopics is an int
# returns an array of array with top words for all topics
top2VecModel = None

def runTop2Vec(docs, numTopics):
    tempTop2VecModel = Top2Vec(docs)
    top2VecModel = tempTop2VecModel.hierarchical_topic_reduction(numTopics)

# get topic words for all topics
def getTopicWords(model):
    return model.topic_words_reduced

# print all wordclouds
def printWordCloud(model, numTopic):
    for i in range(numTopic + 1):
        Top2Vec.generate_topic_wordcloud(
            model, i, background_color="black", reduced=True)

################################################################
#                           BERTopic                          #
################################################################
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
    return mode.get_topic_freq()

def visualize_barchart_bert(model):
    """
        Returns a Figure object that I assume plots out the barchart, 
        Doesnt plot on CLI 
    """
    return model.visualize_barchart()
