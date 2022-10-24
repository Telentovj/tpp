import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.feature_extraction.text import CountVectorizer

def _get_count_vectorizer():
  cv_params = {}
  return CountVectorizer(**cv_params)

def _cosine_similarity(a_i, a_j):
  cs = np.dot(a_i, a_j) / (np.linalg.norm(a_i) * np.linalg.norm(a_j))
  return cs

def _cosine_distance(a_i, a_j):
  cs = _cosine_similarity(a_i, a_j)
  cd = 1 - cs
  return cd

def run_representative_sample_test(docs, doc_topic_df, sample_size=1000, penalty=0):
  '''
  Parameters
  ----------
  docs : np.array
      An array of documents. Note that each document is a string of the processed text.
  
  doc_topic_df : pd.DataFrame
      NOTE: a dataframe which must include 3 columns:
        - 'topic' containing the topic labels
        - 'topic_score' containing the score of the topic for the doc
        - 'doc' containing the unique doc index that maps to docs
  
  Returns
  ----------
  repr_pct : float
      The final metric in percentage terms on how representative are the topic samples.
  '''
  cv = _get_count_vectorizer()
  docs_cv = cv.fit_transform(docs)

  original_word_ct = pd.DataFrame(pd.DataFrame(docs_cv.todense()).sum(axis=0))
  original_word_ct.columns = ['count']
  
  original_index_word_map = pd.DataFrame({'word': count_vectorizer.vocabulary_.keys()}, index=count_vectorizer.vocabulary_.values()).sort_index()

  original_index_word_count_map = pd.merge(original_index_word_map, original_word_ct, how='left', left_index=True, right_index=True)
  original_index_word_count_map.columns = ['original_word', 'original_word_count']

  topics_cd_list = []

  df = deepcopy(doc_topic_df)

  for topic_label in df.topic.unique():
    # Get sample of docs for this topic
    this_topic_df = df[df.topic == topic_label]
    this_topic_docs = docs[list(this_topic_df.doc)]
    this_topic_docs_sample = np.random.choice(this_topic_docs, size=sample_size, replace=False)

    docs_sample_cv = cv.fit_transform(this_topic_docs_sample)

    sample_word_ct = pd.DataFrame(pd.DataFrame(docs_sample_cv.todense()).sum(axis=0))
    sample_word_ct.columns = ['count']

    sample_index_word_map = pd.DataFrame({'word': count_vectorizer.vocabulary_.keys()}, index=count_vectorizer.vocabulary_.values()).sort_index()

    sample_index_word_count_map = pd.merge(sample_index_word_map, sample_word_ct, how='left', left_index=True, right_index=True)
    sample_index_word_count_map.columns = ['sample_word', 'sample_word_count']

    tmp = pd.merge(original_index_word_count_map, sample_index_word_count_map, how='left', left_on='original_word', right_on='sample_word')

    tmp.sample_word_count = tmp.sample_word_count.fillna(penalty)

    # Get comparison arrays and compute cosine distance
    original_v = np.asarray(tmp.original_word_count)
    sample_v = np.asarray(tmp.sample_word_count)
    
    this_topic_cd = _cosine_distance(original_v, sample_v)
    
    topics_cd_list.append(this_topic_cd)
  
  # Get avg cosine distance
  final_cd = sum(topics_cd_list) / len(topics_cd_list)

  # Final representative level metric as a pct
  repr_pct = ((2-final_cd) / 2) * 100

  return repr_pct