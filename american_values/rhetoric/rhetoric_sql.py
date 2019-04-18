import psycopg2 as pg2
import pandas as pd
import string

conn = pg2.connect(dbname= "rhetoric_capstone", host = "localhost")
cur = conn.cursor()
# SQL Pull
query = """SELECT word,
                  lemma,
                  pos,
                  president
            FROM presidents_words"""

# Change line 17 for a new year                                    

pos_df  = pd.read_sql(query, con=conn)

conn.close()

def president_urgency(df):
    forceful_words = 0
    suggestive_words = 0
    possible_words = 0
    questionable_words = 0
    df['word'] = df['word'].str.lower()
    df = df[df['president'] == 'Donald Trump']
    df_md = df[df['pos'] == 'MD']
    df_vbg = df[df['pos'] == 'VBG']

    print(len(df))

    forceful_words += len(df_md[df_md['word'] == 'will'])
    print(df_md[df_md['word'] == 'will'])
    forceful_words += len(df_md[df_md['word'] == 'must'])
    forceful_words += len(df_md[df_md['word'] == 'would'])
    forceful_words += len(df_md[df_md['word'] == 'wo'])
    forceful_words += len(df_md[df_md['word'] == 'need'])
    forceful_words += len(df_vbg[df_vbg['word'] == 'going'])
    forceful_words += len(df_md[df_md['word'] == 'Will'])


    suggestive_words += len(df_md[df_md['word'] == 'should'])
    suggestive_words += len(df_md[df_md['word'] == 'ought'])
    suggestive_words += len(df_md[df_md['word'] == 'shall'])

    possible_words += len(df_md[df_md['word'] == 'can'])
    possible_words += len(df_md[df_md['word'] == 'could'])
    possible_words += len(df_md[df_md['word'] == 'ca'])

    questionable_words += len(df_md[df_md['word'] == 'might'])
    questionable_words += len(df_md[df_md['word'] == 'may'])



    print('Forceful: ', forceful_words)
    print('Suggestive: ', suggestive_words)
    print('Possible: ', possible_words)
    print('Questionable: ', questionable_words)
    

president_urgency(pos_df)

def md_unique_values(df):
    df = df[df['pos'] == 'MD']
    print(df['lemma'].unique())

#md_unique_values(pos_df)

def check_word(df, word):
    df = df[df['word'] == word]
    print(df['pos'].unique())

#check_word(pos_df, 'believe')


def check_pos(df):
    df_1 = df[df['pos'] == "''"]
    print("'' POS: ", df_1['word'].unique())
    df_2 = df[df['pos'] == "$"]
    print("$ POS: ", df_2['word'].unique())
    df_3 = df[df['pos'] == ":"]
    print(": POS: ", df_3['word'].unique())
    df_4 = df[df['pos'] == "``"]
    print("`` POS: ", df_4['word'].unique())
#check_pos(pos_df)

def no_punc_df(df):
    
    print(len(df))
    df = df.dropna()
    df = df[(df['pos'] != "''") | (df['pos'] != "$") | (df['pos'] != "$")]
    print(len(df1))

#no_punc_df(pos_df)