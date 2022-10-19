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
processed_docs_tokenized = [x.split() for x in processed_docs]


docs = list(processed_docs)

################################################################
#                         WordCloud                            #
################################################################

def wordcloud(df):
  comment_words = ''

  for val in df:
      
    comment_words += " ".join(val)+" "
  
  wordcloud = WordCloud(width = 800, height = 800,
                  background_color ='white',
                  stopwords = gensim.parsing.preprocessing.STOPWORDS,
                  min_font_size = 10).generate(comment_words)
  
  # plot the WordCloud image                      
  plt.figure(figsize = (8, 8), facecolor = None)
  plt.imshow(wordcloud)
  plt.axis("off")
  plt.tight_layout(pad = 0)
  
  plt.show()
  return wordcloud

################################################################
#                             NMF                              #
################################################################
### NOTE: Important, need to convert the processed docs to array before
# inputting into NMF

docs_arr = np.asarray(docs)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

def plot_top_words(model, feature_names, n_top_words, title):
    '''
    Parameters
    ----------
    model : sklearn.estimator
        The fitted nmf estimator.

    feature_names : np.array
        The feature names used for training (Selected by TF-IDF Vectorizer).
    
    n_top_words : int
        The number of top words to show for each topic in plot.
    
    title : str
        The main title of the plot.
    '''
    fig, axes = plt.subplots(1, 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 30})
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()

def run_nmf(docs, num_topics):
    '''
    Parameters
    ----------
    docs : np.array
        An array of documents. Note that each document is a string of the processed text.

    num_topics : int
        Number of topics to learn.
    
    Returns
    ----------
    nmf : sklearn.estimator
        The fitted nmf sklearn estimator instance.
    '''
    tfidf_params = {'min_df': 0.0008, 
                    'max_df': 0.90, 
                    'max_features': 500, 
                    'norm': 'l1'}
    nmf_params = {'n_components': num_topics, 
                'alpha_W': 3.108851387228361e-05, 
                'alpha_H': 8.312434671077156e-05, 
                'l1_ratio': 0.3883534426209613, 
                'beta_loss': 'kullback-leibler', 
                'init': 'nndsvda', 
                'solver': 'mu', 
                'max_iter': 1000, 
                'random_state': 4013, 
                'tol': 0.0001}
    
    tfidf_vectorizer = TfidfVectorizer(**tfidf_params)
    tfidf = tfidf_vectorizer.fit_transform(docs)

    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

    nmf = NMF(**nmf_params)
    nmf.fit(tfidf)

    '''
    W is the Document-Topic matrix. 
    Each row in W represents the Document and the entries represents the Document's rank in a Topic.
    H is the Topic-Word matrix (weighting). 
    Each column in H represents a Word and the entries represents the Word's rank in a Topic.
    Matrix multiplication of the factored components, W x H results in the input Document-Word matrix.
    '''

    W = nmf.fit_transform(tfidf)
    H = nmf.components_

    ### Plot top words for each topic
    plot_top_words(nmf, tfidf_feature_names, 10, "Topics in NMF model (KL Divergence Loss)")

    return nmf

################################################################
#                             LDA                              #
################################################################
# !pip install pyLDAvis -qq
import pyLDAvis.gensim_models

def lda(df):

  dictionary = gensim.corpora.Dictionary(df)
  dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
  bow_corpus = [dictionary.doc2bow(doc) for doc in df]
  lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=5, id2word=dictionary, passes=2, workers=2)

  return [lda_model,bow_corpus]

def topic_term(lda):
    for idx, topic in lda.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))

def lda_display(lda_model, bow_corpus, dictionary):
    
    pyLDAvis.enable_notebook()# Visualise inside a notebook
    lda_display = pyLDAvis.gensim_models.prepare(lda_model, bow_corpus, dictionary)
    pyLDAvis.display(lda_display)


################################################################
#                           TOP2VEC                            #
################################################################
from top2vec import Top2Vec

#docs is a string array of documents, numTopic is an integer
def runTop2Vec(docs):
    return Top2Vec(docs)

def runTop2VecReduced(model, numTopics):
     return model.hierarchical_topic_reduction(numTopics)

# get topic words for all topics
def getTopicWords(model):
    return model.topic_words_reduced

# print all wordclouds
def printWordCloud(model, numTopic):
    for i in range(numTopic):
        Top2Vec.generate_topic_wordcloud(
            model, i, background_color="black", reduced=True)
        
import seaborn as sns
#print topic word score barchart
def printWordBar(model, numTopic):
  for i in range(numTopic):
    topic_names = model.topic_words_reduced[i]#[:5]
    topic_probs = model.topic_word_scores_reduced[i]#[:5]
    df_topics = pd.DataFrame(topic_names).rename(columns={0 : "Topic Words"})
    df_probs = pd.DataFrame(topic_probs).rename(columns={0 : "Probability"})
    df = pd.concat([df_topics, df_probs], axis=1)
    plt.figure()
    sns.set(rc={'figure.figsize':(11.7,12)}) 
    ax = sns.barplot(x="Probability", y="Topic Words", data=df, palette="mako").set(title='Topic ' + str(i))

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
    return model.get_topic_freq()

def visualize_barchart_bert(model):
    """
        Returns a Figure object that I assume plots out the barchart, 
        Doesnt plot on CLI 
    """
    return model.visualize_barchart()
