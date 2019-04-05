import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from sklearn import preprocessing
import codecs
import re

import spacy
from spacy.lang.en import English
from spacy import displacy
nlp = spacy.load('en_core_web_md')
from IPython.display import HTML
import logging
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

os.environ['KMP_DUPLICATE_LIB_OK']='True'

with codecs.open('inaug_speeches.csv', 'r', encoding='utf-8', errors='ignore') as fdata:
    df = pd.read_csv(fdata)
print(df)

url = "https://tfhub.dev/google/elmo/2"
embed = hub.Module(url)

text = df.iloc[0].text + ' ' + df.iloc[2].text

text = text.lower().replace('\n', ' ').replace('\t', ' ').replace('\xa0',' ')
text = ' '.join(text.split())
doc = nlp(text)

sentences = []
for i in doc.sents:
  if len(i)>1:
    sentences.append(i.string.strip())
    
len(sentences)

embeddings = embed(
    sentences,
    signature="default",
    as_dict=True)["default"]


with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  sess.run(tf.tables_initializer())
  x = sess.run(embeddings)



pca = PCA(n_components=5)
y = pca.fit_transform(x)


y = TSNE(n_components=2).fit_transform(y)


init_notebook_mode(connected=True)


data = [
    go.Scatter(
        x=[i[0] for i in y],
        y=[i[1] for i in y],
        mode='markers',
        text=[i for i in sentences],
    marker=dict(
        size=16,
        color = [len(i) for i in sentences], #set color equal to a variable
        opacity= 0.8,
        colorscale='Viridis',
        showscale=False
    )
    )
]
layout = go.Layout()
layout = dict(
              yaxis = dict(zeroline = False),
              xaxis = dict(zeroline = False)
             )
fig = go.Figure(data=data, layout=layout)
file = plot(fig, filename='graph.html')

from google.colab import files
files.download('graph.html') 