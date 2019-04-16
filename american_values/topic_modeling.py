from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import codecs
import pandas as pd
from nltk.corpus import stopwords, wordnet 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

    nmf = NMF(n_components=no_topics, random_state=42, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
    transform = nmf.transform(tfidf)
    display_topics(nmf, tfidf_feature_names, no_words)

    return transform 

def clean_scraped_df(df):
    new_df = df[~df['speech_title'].str.contains("Debate")]
    new_df = new_df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
    return new_df

def president_speech_topics(president, no_topics=5, no_words=5, stop_words='english'):
    president_df = df[df['president'] == president]
    speeches = president_df['speech_title'].unique()
    
    for speech in speeches:
        paragraphs = []
        speech_df = president_df[president_df['speech_title'] == speech]
        print(speech)
        for idx, p in speech_df['text'].iteritems():
            p = replace_filler(p)
            p = p.encode('ascii', 'ignore')
            paragraphs.append(p)
        
        nmf_topics(paragraphs, no_topics, no_words, stop_words)

def president_total_topics(president, topics_list, no_topics=5, no_words=15, stop_words='english'):
    president_df = df[df['president'] == president]
    president_df = president_df[~president_df['speech_title'].str.contains("Debate")]
    speeches = president_df['speech_title'].unique()

    all_speeches = []
    for speech in speeches:
        current_speech = []
        speech_df = president_df[president_df['speech_title'] == speech]
        for idx, p in speech_df['text'].iteritems():
            p = replace_filler(str(p))
            current_speech.append(p)
        current_speech = ''.join(current_speech)
        current_speech = current_speech.encode('ascii', 'ignore')
        all_speeches.append(current_speech)
    nmf_matrix = nmf_topics(all_speeches, no_topics, no_words, stop_words)
    topics_df = pd.DataFrame(data=nmf_matrix,    # values
                            index=speeches,    # 1st column as index
                            columns=topics_list)
    print(topics_df)
    print(len(topics_df))
    print(topics_df['6'].sum())
    #president_topics_table(president, topics_list, nmf_matrix, speeches)


def all_presidents(df, no_topics=20, no_words=5, stop_words='english'):
    presidents = df['president'].unique()

    all_speeches = []
    for president in presidents:
        president_df = df[df['president'] == president]
        president_df = president_df[~president_df['speech_title'].str.contains("Debate")]
        speeches = president_df['speech_title'].unique()
        current_president = []
        for speech in speeches:
            current_speech = []
            speech_df = president_df[president_df['speech_title'] == speech]
            for idx, p in speech_df['text'].iteritems():
                p = replace_filler(str(p))
                current_speech.append(p)
            current_speech = ''.join(current_speech)
            current_president.append(current_speech)
        current_president = ''.join(current_president)
        current_president = current_president.encode('ascii', 'ignore')
        all_speeches.append(current_president)
    nmf_matrix = nmf_topics(all_speeches, no_topics, no_words, stop_words)


def president_topics_table(president, topics_list, nmf_matrix, speeches):
    topics_df = pd.DataFrame(data=nmf_matrix,    # values
                             index=speeches,    # 1st column as index
                             columns=topics_list)
    
    sql_df = pd.DataFrame(columns=['president', 'topic', 'score'])
    num_speeches = len(speeches)
    topics_df = topics_df.groupby(topics_df.columns, axis=1).sum()
    try:
        topics_df.drop(columns='Unclear Topic')
    except:
        pass
    counter = 0
    for topic in topics_df.columns:
        
        zeros = len(topics_df[topics_df[topic] == 0])
        total = topics_df[topic].sum()
        score = total * ((num_speeches - zeros) / num_speeches)
        if score == 0:
            pass
        else:
            sql_df.loc[counter] = [president, topic, score]
        counter += 1
    print(topics_df)
    print(sql_df)
    graph_president_topics(sql_df[['topic', 'score']])      


def graph_president_topics(df):
    df = df.set_index('topic')
    df.plot.pie(y='score', labels=None)
    plt.show()


def replace_filler(text):

    text = text.replace('xa0', '')
    text = text.replace('(Applause.)', '')
    text = text.replace('(APPLAUSE.)', '')
    text = text.replace('(applause.)', '')
    text = text.replace('(Laughter.)', '')
    text = text.replace('(LAUGHTER.)', '')
    text = text.replace('(laughter.)', '')
    text = text.replace('(Laughter and applause.)','')
    text = text.replace('(Laughter and Applause.)','')
    text = text.replace('(Applause)', '')
    text = text.replace('(APPLAUSE)', '')
    text = text.replace('(applause)', '')
    text = text.replace('(Laughter)', '')
    text = text.replace('(LAUGHTER)', '')
    text = text.replace('(laughter)', '')
    text = text.replace('(Laughter and applause)','')
    text = text.replace('(Laughter and Applause)','')

    return text

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
    "yourselves", "audience", "thank", "applause", "president", "mr", "booo", "lady"
    "yes", "member", "everybody", "make" "members", "secretary", "shall", "thats", "theyre",
    "just", "ms", "ve", "000", "fellow", "got", "ive" ,"okay", "allen", "150", "lets", "sure",
    "im", "think", "going", "lot", "90", "200", "let",
    ])

topics_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

print(president_total_topics('John Tyler', topics_list, no_topics=10, no_words=10, stop_words=stop_words))


