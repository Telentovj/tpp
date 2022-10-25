import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

### NOTE: Important, need to convert the processed docs to array before
# inputting into NMF

# docs_arr = np.asarray(docs)


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
    fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
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
    return fig

def get_tfidf_vectorizer():
    tfidf_params = {'min_df': 0.0008, 
                    'max_df': 0.90, 
                    'max_features': 500, 
                    'norm': 'l1'}
    return TfidfVectorizer(**tfidf_params)

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
    tfidf_feature_names: list[str]
        Vocabulary to aid visualisation.
    '''
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
    
    tfidf_vectorizer = get_tfidf_vectorizer()
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

    return nmf, tfidf_feature_names

def get_doc_topic_df(docs, model, num_topics):
    '''
    Parameters
    ----------
    docs : np.array
        An array of documents. Note that each document is a string of the processed text.
        This is same as input into run_nmf.
    
    model : sklearn.estimator
        The fitted nmf estimator returned from run_nmf. 

    num_topics : int
        Number of topics learned.
    
    Returns
    ----------
    doc_topic_df : pd.DataFrame
        df with 3 columns, topic, topic_score and doc which are 
        the topic labels, scores and document index respectively.
    '''
    tfidf_vectorizer = get_tfidf_vectorizer()
    tfidf = tfidf_vectorizer.fit_transform(docs)

    W = model.fit_transform(tfidf)
    W = pd.DataFrame(W)

    W['topic'] = W.apply(lambda r: r.argmax(), axis=1)
    W['topic_score'] = W.apply(lambda r: r[:num_topics].max(), axis=1)
    W['doc'] = W.index

    doc_topic_df = W[['topic', 'topic_score', 'doc']]

    return doc_topic_df

def get_top_docs_nmf(df, docs, model, num_topics, k):
    '''
    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe with 'text' column.
    
    docs : np.array
        An array of documents. Note that each document is a string of the processed text.
        This is same as input into run_nmf.
    
    model : sklearn.estimator
        The fitted nmf estimator returned from run_nmf. 

    num_topics : int
        Number of topics learned.
    
    k : int
        The top k number of docs will be taken from each topic's docs.
    
    Returns
    ----------
    top_k_docs : list
        Array of top scoring sample docs for the topics.
    '''
    doc_topic_df = get_doc_topic_df(docs, model, num_topics)

    docs_idx = []

    for topic in doc_topic_df.topic.unique():
        top_k_docs_df = doc_topic_df[doc_topic_df.topic == topic].sort_values('topic_score', ascending=False)[:k]
        docs_idx.extend(list(top_k_docs_df.doc))

    top_k_docs = list(df.loc[docs_idx, 'text'].values[0])

    return top_k_docs