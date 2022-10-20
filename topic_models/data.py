import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer 
from nltk.stem.porter import *
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('wordnet')
nltk.download('omw-1.4’)

################################################################
#                         LOADING DATA                         #
################################################################
def load_data(file, ):
    """
    Args:
    - file: Streamlit File Object or String for file path

    Returns:
    - Pandas dataframe
    """
    df = pd.read_csv(file)
    return df

################################################################
#                      DATA PREPROCESSING                      #
################################################################
def _clean_data(df):
    """
    Preprocesses data by removing na values, words associated with @ and # 
    as well as http[s] links

    Args:
    - df: pandas dataframe

    Returns:
    - Cleaned text in the form of a pandas dataframe with row number and text
    """
    df["text"] = df["text"].str.lower()
    df["text"] = df["text"].replace("na", None)
    text = df[~df["text"].isna()][['text']]

    # Remove @, # and links
    text["text"] = text["text"].str.replace(r"@[A-Za-z0-9_]+",'', regex=True)
    text["text"] = text["text"].str.replace(r"#[A-Za-z0-9_]+",'', regex=True)
    text["text"] = text["text"].str.replace(r"http[s]?://\S+",'', regex=True)

    return text

def _lemmatize(text):
    return WordNetLemmatizer().lemmatize(text, pos='v')

def _preprocess(text):
    """
    Args:
    - text: String

    Returns:
    - Cleaned text: String
    """
    result = ''
    text=str(text)
    token_words = gensim.utils.simple_preprocess(text)
    for token in token_words:
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result = result + ' ' + _lemmatize(token)
    return result

def preprocess_data(df):
    """
    Args:
    - df: pandas dataframe

    Returns:
    - text: pandas dataframe and pair of two arrays docs and docs_tokenized
        - docs: list of strings (sentences)
        - docs_tokenized: list of tokenized words (bag of words)
    """
    text = _clean_data(df)
    text["processed"] = text["text"].map(_preprocess)
    docs_tokenized = [x.split() for x in text["processed"]]
    docs = list(text["processed"])

    return (text, docs, docs_tokenized)

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
#                       HELPER FUNCTIONS                       #
################################################################
def samples_to_csv(samples):
    """
    Converts list of samples to output an encoded csv for streamlit
    """
    df = pd.DataFrame(samples, columns=["text"])
    return df.to_csv().encode('utf-8')