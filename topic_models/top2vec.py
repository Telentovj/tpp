import pandas as pd
from top2vec import Top2Vec
import matplotlib.pyplot as plt
import seaborn as sns

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

def get_top_documents_Top2Vec(df, model, num_topics, k):
    samples = []
    for topic_num in range(0, num_topics):
        documents = documents, document_scores, document_ids = model.search_documents_by_topic(topic_num = topic_num, num_docs = k, reduced = True)
        for index, doc in enumerate(documents):
            if index > k:
                break
            sample = df.loc[df['processed'] == doc]['text'].values[0]
            samples.append(sample)
    return samples