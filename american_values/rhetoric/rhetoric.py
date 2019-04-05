import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import psycopg2 as pg2
from rhetoric_dataset import dataset_query
import numpy as np

inaug_df = dataset_query()

def pos_unique_values(df, pos):
    pos = pos.upper() # Make sure part of speech is uppercase
    unique_values = df[df['pos'] == pos]['lemma'].unique() # Find unique values for specific part of speech

    return unique_values

def plot_lines(df, title):
    df = df[df['category'] != 'n/a']
    filepath = 'graphs/' + title
    sns.lineplot(x=df['year'], y=df["freq"], hue=df['category'], ci=None) 
    plt.title(title)
    plt.savefig(filepath)
    plt.show()

def line_plots(df, title_add=""):
    num_categories = len(df['category'].unique())
    df = df[df['category'] != 'n/a']
    for i in range(num_categories):
        filepath = 'graphs/' + title_add + ' ' + df['category'].unique()[i]
        category_df = df[df['category'] == df['category'].unique()[i]]
        category_df = category_df.groupby(['year', 'category']).sum()
        sns.lineplot(x=category_df.index.levels[0], y=category_df["freq"])
        plt.title(df['category'].unique()[i])
        plt.savefig(filepath)
        plt.show()

def MD_graph_df(df):
    graph_df = pd.DataFrame(columns=['year', 'word', 'freq']) #Create empty new DF for graph
    year_count = df.groupby('year').count() # Find Length of X axis for graph
    counter = 0
    pos = 'MD'
    word_list = pos_unique_values(df, pos) # Find all unique values for pos
    for i in range(len(year_count)): #For every year
        pos_df = df[df['pos'] == pos] # Filter dataframe to specific part of speech
        pos_df = pos_df[pos_df['year'] == year_count.index[i]].groupby(['lemma', 'pos']).size() # Count occurences of each lemma with that part of speech for the current year in the for loop
        total_words = year_count.iloc[i, 1] # Find total words in year
        year = year_count.index[i] # Find year
        
        for word in word_list: # For every unique word of the modal part of speech tab
            if word in pos_df.index.levels[0]:
                freq = pos_df[word][0] / total_words # Determine frequency of word in speech
            else:
                freq = 0 # If word not in speech assign freqeuncy value of 0
            graph_df.loc[counter] = [year, word, freq]
            counter += 1  

    grouped = []
    for idx, word in graph_df['word'].iteritems(): # Group similar values together within dataframe
        if word == 'will' or word == 'would' or word == 'need' or word == 'must':
            grouped.append('forceful_language')
        elif word == 'should' or word == 'shall' or word == 'ought':
            grouped.append('suggestive_language')
        elif word == 'could' or word == 'can':
            grouped.append('probable_language')
        elif word == 'might' or word == 'may':
            grouped.append('possible_language')
        else:
            grouped.append('n/a')
    graph_df['category'] = grouped #Add category column to dataframe

    return graph_df

def PRP_graph_df(df):
    graph_df = pd.DataFrame(columns=['year', 'word', 'freq']) #Create empty new DF for graph
    year_count = df.groupby('year').count() # Find Length of X axis for graph
    counter = 0
    word_list = np.append(pos_unique_values(df, 'PRP'), pos_unique_values(df, 'PRP$')) # Find all unique values for pos
    for i in range(len(year_count)): #For every year
        pos_df = df[(df['pos'] == 'PRP$') | (df['pos'] == 'PRP')] # Filter dataframe to specific part of speech
        pos_df = pos_df[pos_df['year'] == year_count.index[i]].groupby(['lemma', 'pos']).size() # Count occurences of each lemma with that part of speech for the current year in the for loop
        total_words = year_count.iloc[i, 1] # Find total words in year
        year = year_count.index[i] # Find year
        
        for word in word_list: # For every unique word of the modal part of speech tab
            if word in pos_df.index.levels[0]:
                freq = pos_df[word][0] / total_words # Determine frequency of word in speech
            else:
                freq = 0 # If word not in speech assign freqeuncy value of 0
            graph_df.loc[counter] = [year, word, freq]
            counter += 1  

    grouped = []
    for idx, word in graph_df['word'].iteritems(): # Group similar values together within dataframe
        if word in ['me', 'i', 'myself', 'my']:
            grouped.append('Me')
        elif word in ['yours', 'yourself', 'your', 'you']:
            grouped.append('You')
        elif word in ['they', 'themselves', 'them', 'theirs', 'their']:
            grouped.append('They')
        elif word in ['he', 'himself', 'him', 'she', 'her', 'herself', 'her', 'his']:
            grouped.append('He_She')
        elif word in ['we', 'us', 'ourselves', 'our']:
            grouped.append('We')
        else:
            grouped.append('n/a')
    graph_df['category'] = grouped #Add category column to dataframe

    return graph_df



def gender_df(df):
    graph_df = pd.DataFrame(columns=['year', 'word', 'freq']) #Create empty new DF for graph
    year_count = df.groupby('year').count() # Find Length of X axis for graph
    counter = 0
    word_list = np.append(pos_unique_values(df, 'PRP'), pos_unique_values(df, 'PRP$')) # Find all unique values for pos
    for i in range(len(year_count)): #For every year
        pos_df = df[(df['pos'] == 'PRP$') | (df['pos'] == 'PRP')] # Filter dataframe to specific part of speech
        pos_df = pos_df[pos_df['year'] == year_count.index[i]].groupby(['lemma', 'pos']).size() # Count occurences of each lemma with that part of speech for the current year in the for loop
        total_words = year_count.iloc[i, 1] # Find total words in year
        year = year_count.index[i] # Find year
        
        for word in word_list: # For every unique word of the modal part of speech tab
            if word in pos_df.index.levels[0]:
                freq = pos_df[word][0] / total_words # Determine frequency of word in speech
            else:
                freq = 0 # If word not in speech assign freqeuncy value of 0
            graph_df.loc[counter] = [year, word, freq]
            counter += 1  

    grouped = []
    for idx, word in graph_df['word'].iteritems(): # Group similar values together within dataframe
        if word in ['he', 'himself', 'him', 'his']:
            grouped.append('He')
        elif word in ['she', 'her', 'herself', 'her']:
            grouped.append('She')
        else:
            grouped.append('n/a')
    graph_df['category'] = grouped
    
    return graph_df

def determiners_df(df):
    graph_df = pd.DataFrame(columns=['year', 'word', 'freq']) #Create empty new DF for graph
    year_count = df.groupby('year').count() # Find Length of X axis for graph
    counter = 0
    pos = 'DT'
    word_list = pos_unique_values(df, pos) # Find all unique values for pos
    for i in range(len(year_count)): #For every year
        pos_df = df[df['pos'] == pos] # Filter dataframe to specific part of speech
        pos_df = pos_df[pos_df['year'] == year_count.index[i]].groupby(['lemma', 'pos']).size() # Count occurences of each lemma with that part of speech for the current year in the for loop
        total_words = year_count.iloc[i, 1] # Find total words in year
        year = year_count.index[i] # Find year
        
        for word in word_list: # For every unique word of the modal part of speech tab
            if word in pos_df.index.levels[0]:
                freq = pos_df[word][0] / total_words # Determine frequency of word in speech
            else:
                freq = 0 # If word not in speech assign freqeuncy value of 0
            graph_df.loc[counter] = [year, word, freq]
            counter += 1  

    grouped = []
    for idx, word in graph_df['word'].iteritems(): # Group similar values together within dataframe
        if word in ['every', 'all', 'any', 'each']:
            grouped.append('Outstanding Hyperbole')
        elif word == 'some':
            grouped.append('Middling Hyperbole')
        else:
            grouped.append('n/a')
    graph_df['category'] = grouped #Add category column to dataframe

    return graph_df

#print(pos_unique_values(inaug_df, 'DT'))
#print(pos_unique_values(inaug_df, 'PDT'))

graph_df = determiners_df(inaug_df)
plot_lines(graph_df, 'Hyperbole')

'''
GROUP = nation, state, country, union, national
GROUP = world vs foreign
GROUP = peace
GROUP = america, american
GROUP = new, change, progress, 
GROUP = war, strength
GROUP = law, justice, principle
GROUP = life
GROUP = freedom, free, liberty, 
GROUP = duty, work, responsibility, effort, honor
GROUP = hope, good, better
GROUP = make,
GROUP = god, faith
GROUP = know, never, best
GROUP = history, old
GROUP = future
GROUP = question
GROUP = republic, 
GROUP = necessary
'''

['every', 'all', 'either', 'another', 'any', 'each']