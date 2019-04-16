import pandas as pd
import codecs
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from topic_modeling import replace_filler
import psycopg2 as pg2
from nltk.tokenize import RegexpTokenizer
from statistics import mean



with codecs.open('data/speeches.csv', 'r', encoding='utf-8', errors='ignore') as fdata:
    df = pd.read_csv(fdata)

def create_stats_table(df):
    presidents = df['president'].unique()

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
            current_speech = current_speech.encode('ascii', 'ignore')
            current_president.append(current_speech)

        tfidf_vectorizer = TfidfVectorizer(stop_words=None)
        tfidf = tfidf_vectorizer.fit_transform(current_president)
        tfidf_feature_names = tfidf_vectorizer.get_feature_names()

        print('President: ', president)
        print('Speeches: ', len(speeches))
        print('Vocab: ', len(tfidf_vectorizer.vocabulary_))

#create_stats_table(df)

def president_total_words(df):
    presidents = df['president'].unique()
    stats_df = pd.DataFrame(columns=['president_num','president', 'avg_vocab', 'avg_words', 'total_vocab', 'total_words', 'total_speeches'])
    pres_id = 44
    counter = 0
    for president in presidents:
        print(president)
        president_df = df[df['president'] == president]
        president_df = president_df[~president_df['speech_title'].str.contains("Debate")]
        speeches = president_df['speech_title'].unique()
        current_president = []
        speeches_vocab = []
        speeches_words = []
        for speech in speeches:
            current_speech = []
            speech_df = president_df[president_df['speech_title'] == speech]
            for idx, p in speech_df['text'].iteritems():
                p = replace_filler(str(p))
                current_speech.append(p)
            tfidf_vectorizer_speech = TfidfVectorizer(stop_words=None)
            tfidf_speech = tfidf_vectorizer_speech.fit_transform(current_speech)
            speeches_vocab.append(len(tfidf_vectorizer_speech.vocabulary_))

            current_speech = ''.join(current_speech)

            tokenizer = RegexpTokenizer(r'\w+')
            tokens = tokenizer.tokenize(current_speech)
            speeches_words.append(len(tokens))

            current_speech = current_speech.encode('ascii', 'ignore')
            current_president.append(current_speech)

        tfidf_vectorizer_president = TfidfVectorizer(stop_words=None)
        tfidf_president = tfidf_vectorizer_president.fit_transform(current_president)
        total_vocab = len(tfidf_vectorizer_president.vocabulary_)

        stats_df.loc[counter] = [pres_id, president, mean(speeches_vocab), mean(speeches_words), total_vocab, sum(speeches_words), len(speeches)]
        counter += 1
        pres_id -= 1

    print(stats_df)

#president_total_words(df)


def crowd_reactions(df):
    presidents = df['president'].unique()

    for president in presidents:
        print(president)
        president_df = df[df['president'] == president]
        president_df = president_df[~president_df['speech_title'].str.contains("Debate")]
        speeches = president_df['speech_title'].unique()
        for speech in speeches:
            current_speech = []
            speech_df = president_df[president_df['speech_title'] == speech]
            applause = 0
            laughter = 0
            for idx, p in speech_df['text'].iteritems():
                current_speech.append(p)
            current_speech = ''.join(current_speech)
            applause += current_speech.split().count('(Applause.)')
            applause += current_speech.split().count('(APPLAUSE.)')
            applause += current_speech.split().count('(applause.)')
            applause += current_speech.split().count('(Applause)')
            applause += current_speech.split().count('(APPLAUSE)')
            applause += current_speech.split().count('(applause)')
            applause += current_speech.split().count('(Laughter and applause.)')
            applause += current_speech.split().count('(Laughter and Applause.)')
            applause += current_speech.split().count('(Laughter and applause)')
            applause += current_speech.split().count('(Laughter and Applause)')

            laughter += current_speech.split().count('(Laughter)')
            laughter += current_speech.split().count('(LAUGHTER)')
            laughter += current_speech.split().count('(laughter)')
            laughter += current_speech.split().count('(Laughter.)')
            laughter += current_speech.split().count('(LAUGHTER.)')
            laughter += current_speech.split().count('(laughter.)')
            laughter += current_speech.split().count('(Laughter and applause.)')
            laughter += current_speech.split().count('(Laughter and Applause.)')
            laughter += current_speech.split().count('(Laughter and applause)')
            laughter += current_speech.split().count('(Laughter and Applause)')

            print('speech:', speech)
            print('applause: ', applause)
            print('laughter: ', laughter)

crowd_reactions(df)

def forceful_language():   
    conn = pg2.connect(dbname= "rhetoric_capstone", host = "localhost")
    cur = conn.cursor()
    # SQL Pull
    forceful_sort = """SELECT game_id::integer,
                            team_id,
                            player_id,
                            player_name,
                            SUM(wins_contr) AS WINS
                    FROM presidesnts_words
                    GROUP BY game_id, player_id, player_name, team_id
                    ORDER BY game_id"""

    # Change line 17 for a new year                                    

    season_df = pd.read_sql(season_sort, con=conn)

    conn.close()

#print(df[df['speech_title'] == 'Statement on the School Shooting in Parkland, Florida'])