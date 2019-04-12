import pandas as pd
import psycopg2 as pg2
from sqlalchemy import create_engine

df1 = pd.read_csv('final_21500_21700.csv')
df2 = pd.read_csv('final_26000_26200.csv')
df3 = pd.read_csv('final_26400_26600.csv')
df4 = pd.read_csv('final_26600_26800.csv')
df5 = pd.read_csv('final_26800_27000.csv')
df6 = pd.read_csv('final_27000_27200.csv')
df7 = pd.read_csv('final_27200_27500.csv')
df8 = pd.read_csv('final_26200_26400.csv')



dfs = [df1,
df2,
df3,
df4,
df5,
df6,
df7,
df8d]

conn = pg2.connect(dbname = 'postgres', host = "localhost")
conn.autocommit = True
engine = create_engine('postgresql+psycopg2://owner:Fulfyll@localhost/rhetoric_capstone')
print(len(dfs))
for df in dfs:
    print(len(df))
    df.to_sql("presidesnts_words", con = engine, if_exists= "append", index=False)
conn.close()