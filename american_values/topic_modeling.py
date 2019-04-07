from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import codecs
import pandas as pd
from nltk.corpus import stopwords, wordnet 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import re

with codecs.open('data/speeches.csv', 'r', encoding='utf-8', errors='ignore') as fdata:
    df = pd.read_csv(fdata)

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic {}:".format(topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))


def tfidf_fit(documents):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(documents)
    return tfidf



def clean_inaug_speeches(df): 
    documents = []
    for idx, d in df['text'].iteritems():
        d = d.replace('0092', '')
        d = d.replace('0097', '')
        documents.append(d)

    year = int(df.loc[idx]['Date'][-4:])
    return documents, year

def lda_topics(documents, no_topics, no_words):
    # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
    tf_vectorizer = CountVectorizer(stop_words='english')
    tf = tf_vectorizer.fit_transform(documents)
    tf_feature_names = tf_vectorizer.get_feature_names()
     
    lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
    display_topics(lda, tf_feature_names, no_words)
       
def nmf_topics(documents, no_topics, no_words, stopwords):
    # NMF is able to use tf-idf
    tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords)
    tfidf = tfidf_vectorizer.fit_transform(documents)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()

    nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
    display_topics(nmf, tfidf_feature_names, no_words)
    #print(nmf.components_)


def clean_scraped_df(df):
    new_df = df[~df['speech_title'].str.contains("Debate")]
    new_df = new_df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
    return new_df

def president_topics(president, no_topics=5, no_words=5, stop_words='english'):
    president_df = df[df['president'] == president]
    speeches = president_df['speech_title'].unique()
    
    for speech in speeches:
        documents = []
        speech_df = president_df[president_df['speech_title'] == speech]
        print(speech)
        for idx, d in speech_df['text'].iteritems():
            d = d.replace('xa0', '')
            d = d.replace('(Applause)', '')
            d = d.replace('(applause)', '')
            d = d.replace('(Laughter)', '')
            d = d.replace('(laughter)', '')
            d = d.replace('(Laughter and applause)','')
            documents.append(d)
        nmf_topics(documents, no_topics, no_words, stop_words)

#speeches_df = clean_scraped_df(df) 

stop_words = frozenset([
    "a", "about", "above", "across", "after", "afterwards", "again", "against",
    "all", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
    "around", "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both",
    "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con",
    "could", "couldnt", "cry", "de", "describe", "detail", "do", "done",
    "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
    "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill",
    "find", "fire", "first", "five", "for", "former", "formerly", "forty",
    "found", "four", "from", "front", "full", "further", "get", "give", "go",
    "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
    "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
    "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter",
    "latterly", "least", "less", "ltd", "made", "many", "may", "me",
    "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
    "move", "much", "must", "my", "myself", "name", "namely", "neither",
    "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
    "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on",
    "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
    "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
    "please", "put", "rather", "re", "same", "see", "seem", "seemed",
    "seeming", "seems", "serious", "several", "she", "should", "show", "side",
    "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
    "something", "sometime", "sometimes", "somewhere", "still", "such",
    "system", "take", "ten", "than", "that", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
    "third", "this", "those", "though", "three", "through", "throughout",
    "thru", "thus", "to", "together", "too", "top", "toward", "towards",
    "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
    "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself",
    "yourselves", "audience", "thank", "applause", "president", "mr", "booo", "speaker", "lady"
    "yes", "member", "everybody", "make", "vice", "members", "secretary"])


no_topics = 1
no_words = 5

print(president_topics('Abraham Lincoln', no_topics, no_words, stop_words))
