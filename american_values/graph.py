import psycopg2 as pg2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

conn = pg2.connect(dbname= "rhetoric_capstone", host = "localhost")
cur = conn.cursor()
# SQL Pull
pos_query = """SELECT lemma, 
                        year
                 FROM presidesnts_words
                 WHERE pos = 'MD'
                 ORDER BY year"""

                                                  

pos_df = pd.read_sql(pos_query, con=conn)
conn.close()

pos_df['lemma'] = pos_df['lemma'].str.lower()
pos_df = pos_df.groupby(['lemma', 'year']).size()
pos_df = pos_df.unstack(level=0).fillna(value=0)


for j in pos_df.columns:
    graph_series = pos_df[j]
    graph_series.plot.line()
    plt.title(j)
    plt.show()

