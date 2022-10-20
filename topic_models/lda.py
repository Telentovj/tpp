import gensim
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