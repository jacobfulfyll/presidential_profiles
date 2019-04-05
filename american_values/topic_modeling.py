from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import codecs
import pandas as pd
from nltk.corpus import stopwords, wordnet 
from nltk.stem.wordnet import WordNetLemmatizer
import string

with codecs.open('data/inaug_speeches.csv', 'r', encoding='utf-8', errors='ignore') as fdata:
    df = pd.read_csv(fdata)

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic {}:".format(topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))


def tfidf_fit(list):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(documents)
    return tfidf



def clean_speeches(df): 
    documents = []
    for idx, d in df['text'].iteritems():
        d = d.replace('0092', '')
        d = d.replace('0097', '')
        documents.append(d)

    year = int(df.loc[idx]['Date'][-4:])
    return documents, year

def lda_topics(df, no_topics, no_words):
    documents, year = clean_speeches(df)
    # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
    tf_vectorizer = CountVectorizer(stop_words='english')
    tf = tf_vectorizer.fit_transform(documents)
    tf_feature_names = tf_vectorizer.get_feature_names()
     
    lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
    display_topics(lda, tf_feature_names, no_words)
       
def nmf_topics(df, no_topics, no_words):
    documents, year = clean_speeches(df)
    # NMF is able to use tf-idf
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(documents)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()

    nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
    display_topics(nmf, tfidf_feature_names, no_words)
    print(nmf.components_)


no_topics = 5
no_words = 5

print('========== NMF ==========')
print(nmf_topics(df, no_topics, no_words))
