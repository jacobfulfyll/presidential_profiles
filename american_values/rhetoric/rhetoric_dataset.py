import codecs
import pandas as pd
from nltk.corpus import wordnet as wn
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
import string
import psycopg2 as pg2
from sqlalchemy import create_engine

with codecs.open('../data/speeches.csv', 'r', encoding='utf-8', errors='ignore') as fdata:
    df = pd.read_csv(fdata)

lemmatizer = WordNetLemmatizer() # Create Lemmatizer

def create_dataset(df, db, table):
    inaug_df = pd.DataFrame(columns=['word', 'pos', 'lemma', 'year', 'president']) #Create empty DF
    counter = 0
    for idx, doc in df['text'].iteritems(): # For every inauguration speech
        doc = doc.replace('xa0', '')
        doc = doc.replace('(Applause)', '')
        doc = doc.replace('(applause)', '')
        doc = doc.replace('(Laughter)', '')
        doc = doc.replace('(laughter)', '')
        doc = doc.replace('(Laughter and applause)','')
        text = nltk.word_tokenize(doc) # Tokenize all words in current speech
        pos_tags = nltk.pos_tag(text) # Create Part of Speech Tag for every word
        year = int(df.loc[idx]['date'][-4:]) # Pull Year From Date Column, for new DF
        president = df.loc[idx]['president']
        print(idx, ' / ', len(df)) # Status Check
        for item in pos_tags:
            if item[0] in string.punctuation: # Remove Punctuation
                pass
            else:
                word = item[0].lower() # Seperate Word and Part of Speech into different variables
                part_of_speech = item[1] # To be placed in new DF

                # Lemmatize all possible words, adj, verbs, nouns, adverbs
                if str(item[1][0]) == 'J':
                    lemma_tag = 'a'
                    lemma = lemmatizer.lemmatize(item[0], lemma_tag)
                elif item[1][0] == 'V' or item[1][0] == 'N' or item[1][0] == 'R':
                    lemma_tag = str(item[1][0]).lower()
                    lemma = lemmatizer.lemmatize(item[0], lemma_tag)
                else:
                    lemma = item[0]
                
                inaug_df.loc[counter] = [word, part_of_speech, lemma, year, president] # Add row to DF
                counter += 1 # Move to the next row
    
    sql_upload(inaug_df, db, table)

def sql_upload(df, db, table): # Upload a DF to SQL
    conn = pg2.connect(dbname = 'postgres', host = "localhost")
    conn.autocommit = True
    engine = create_engine('postgresql+psycopg2://owner:Fulfyll@localhost/' + db)
    df.to_sql(table, con = engine, if_exists= "append", index=False)
    conn.close()

def dataset_query(): # Create Pandas DF from entire SQL Table
    conn = pg2.connect(dbname="rhetoric_capstone" , host = "localhost")
    # SQL Pull
    query = """SELECT *
               FROM inaug_speeches
            """
                                  

    inaug_df = pd.read_sql(query, con=conn)

    conn.close()

    return inaug_df

create_dataset(df, 'rhetoric_capstone', 'all_speeches')