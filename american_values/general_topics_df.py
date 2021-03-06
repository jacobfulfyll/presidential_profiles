import pandas as pd
from topic_modeling import president_total_topics

def topics_table():
    topics_df = pd.DataFrame(columns=['president_num','president', 'topics_list'])
    topics_df.loc[0] = [44, 'Donald Trump', ['America', 'Internal Tension', 'Immigration', 'Tragedy', 'Internal Tension', 'Natural Resources', 'Drug Use', None, None, None]]
    topics_df.loc[1] = [43, 'Barack Obama', [['America', 'Economy'], 'Foreign Intervention', 'Foreign Intervention', 'Foreign Intervention', 'Legislation', 'Legislation', 'Internal Tension', 'Foreign Intervention', 'Nomination', 'Foreign Diplomacy']]
    topics_df.loc[2] = [42, 'George W. Bush', ['America', 'Foreign Intervention', 'Technology', 'Foreign Intervention', 'Legislation', 'Natural Disaster', 'Economy', 'Immigration', 'Legislation', 'Faith']]
    topics_df.loc[3] = [41, 'Bill Clinton', ['America', 'Foreign Intervention', 'Foreign Intervention', 'Tragedy', 'Foreign Intervention', 'Legislation', 'Foreign Diplomacy', 'Scandal', 'Foreign Diplomacy', 'Natural Disaster']]
    topics_df.loc[4] = [40, 'George H. W. Bush', ['America', 'Foreign Intervention', 'US War', 'Foreign Intervention', 'Legislation', 'Foreign Intervention', 'Legislation', 'Foreign Intervention', 'Foreign Intervention', 'US War']]
    topics_df.loc[5] = [39, 'Ronald Reagan', ['America', 'Foreign Diplomacy', 'Foreign Intervention', 'Legislation', 'Internal Tension', 'Tragedy', 'Foreign Intervention', 'Historical Reference', 'Unclear Topic', 'Drug Use']]
    topics_df.loc[6] = [38, 'Jimmy Carter', ['Natural Resources', 'US War', 'Unclear Topic', 'Foreign Intervention', 'Foreign Intervention', 'Foreign Diplomacy', 'Foreign Intervention', 'Economy', None, None]]
    topics_df.loc[7] = [37, 'Gerald Ford', ['America', 'Natural Resources', 'Values', 'Legislation', 'Scandal', 'America', 'Foreign Intervention', 'Global Cooperation', None, None]]
    topics_df.loc[8] = [36, 'Richard Nixon', ['America', 'Foreign Intervention', 'Scandal', 'Scandal', 'US War', 'Internal Tension', 'Natural Resources', 'Internal Tension','Internal Tension', 'Internal Tension']]
    topics_df.loc[9] = [35, 'Lyndon B. Johnson', [['America', 'Global Cooperation'], 'Foreign Intervention', 'Foreign Intervention', 'Global Cooperation', 'Foreign Intervention', 'Foreign Intervention', 'Technology', 'US War', 'Internal Tension', 'Faith']]
    topics_df.loc[10] = [34, 'John F. Kennedy', ['Global Cooperation', 'Internal Tension', 'Economy', 'Internal Tension', 'News', 'Foreign Intervention', 'US War', 'Unclear Topic', 'Faith', 'Institution Creation']]
    topics_df.loc[11] = [33, 'Dwight D. Eisenhower', ['Global Cooperation', 'Foreign Intervention', 'Unclear Topic', 'Internal Tension', 'Technology', None, None, None, None, None]]
    topics_df.loc[12] = [32, 'Harry S. Truman', ['Global Cooperation', 'Internal Tension', 'US War', 'Technology', 'Internal Tension', 'Internal Tension', 'Foreign Intervention', 'Internal Tension','Foreign Intervention', None]]
    topics_df.loc[13] = [31, 'Franklin D. Roosevelt', [['US War', 'America'], 'Economy', 'Global Cooperation', 'US War', 'US War', 'Economy', 'Legislation', 'US War','Natural Resources','Economy']]
    topics_df.loc[14] = [30, 'Herbert Hoover', ['Economy', 'Foreign Diplomacy', 'Economy', 'Internal Tension', 'Legislation', 'Crime', 'Economy', 'Legislation', 'Global Cooperation', 'Economy']]
    topics_df.loc[15] = [29, 'Calvin Coolidge', ['America', 'Religion', 'Historical Reference','Historical Reference', None,'Historical Reference', None, None, None, None]]
    topics_df.loc[16] = [28, 'Warren G. Harding', ['America', 'Internal Tension', 'Values', 'Historical Reference', 'Internal Tension', 'Internal Tension', 'Values', None, ['America', 'Values'], 'Values']]
    topics_df.loc[17] = [27, 'Woodrow Wilson', ['Global Cooperation', 'US War', 'Global Cooperation', None, ['Historical Reference', 'America'], 'US War', 'Values', 'Foreign Intervention', 'Economy', 'Internal Tension']]
    topics_df.loc[18] = [26, 'William Taft', ['America', 'Economy', 'Economy', 'Legislation', 'Legislation', 'Natural Resources', 'Legislation', None, None, None]]
    topics_df.loc[19] = [25, 'Theodore Roosevelt', ['America', 'America', 'Foreign Intevention', 'Foreign Diplomacy', None, 'Foreign Intervention', 'Foreign Intervention', 'Legislation', 'Internal Tension', 'Foreign Diplomacy']]
    topics_df.loc[20] = [24, 'William McKinley', ['America', 'US War', 'Legislation', 'US War', 'America', 'Economy', None, None, None, None]]
    topics_df.loc[21] = [23, 'Grover Cleveland', ['America', 'Economy', 'Foreign Diplomacy', ['Immigration', 'Internal Tension'], 'Internal Tension', 'Legislation', 'Foreign Intervention', 'Immigration', 'Natural Resources', 'Historical Reference']]
    topics_df.loc[22] = [22, 'Benjamin Harrison', ['America', 'Internal Tension', 'Foreign Intervention', 'Historical Reference', 'Natural Disaster', 'Nomination', 'Foreign Intervention', 'Foreign Intervention', 'America', ['Faith', 'Historical Reference']]]
    topics_df.loc[24] = [20, 'James A. Garfield', ['American Government', None, None, None, None, None, None, None, None, None]]
    topics_df.loc[25] = [19, 'Rutherford B. Hayes', ['America', 'Legislation', 'Foreign Intervention', 'Economy', 'Internal Tension', None, 'Economy', 'Immigration', None, 'Military Policy']]
    topics_df.loc[26] = [18, 'Ulysses S. Grant', ['America', 'Internal Tension', 'Legislation', None, 'Foreign Diplomacy', 'Internal Tension', 'Faith', 'Legislation', 'Legislation','Legislation']]
    topics_df.loc[27] = [17, 'Andrew Johnson', ['America', 'Internal Tension', 'Internal Tension', ['America', 'US War'], 'Internal Tension', 'Internal Tension', 'Internal Tension', 'Crime', 'US War', 'Historical Reference']]
    topics_df.loc[28] = [16, 'Abraham Lincoln', ['America', 'Internal Tension', None, 'Values', ['US War', 'Internal Tension'], ['US War', 'Faith'], 'Values', 'Internal Tension',['Historical Reference', 'Values'], ['Historical Reference', 'Internal Tension']]]
    topics_df.loc[29] = [15, 'James Buchanan', ['America', 'Internal Tension', 'Internal Tension', 'Internal Tension', 'Internal Tension', 'Foreign Intervention', 'Foreign Intervention', 'Unclear Topic', 'Unclear Topic', 'US Expansion']]
    topics_df.loc[30] = [14, 'Franklin Pierce', ['America', None, 'Foreign Intervention', 'Crime', None, 'Military Policy', 'Foreign Diplomacy', 'Foreign Diplomacy', 'Foreign Diplomacy', 'Foreign Intervention']]
    topics_df.loc[31] = [13, 'Millard Fillmore', ['America', 'Foreign Intervention', 'Military Policy',  None, None, None, None, None, None, None]]
    topics_df.loc[32] = [12, 'Zachary Taylor', [['America', 'Foreign Intervention'], 'Foreign Diplomacy', None, None, None, None, None, None, None, None]]
    topics_df.loc[33] = [11, 'James K. Polk', ['US War', 'US Way', 'Internal Tension', 'Foreign Diplomacy', None, 'US War', 'US War', 'Foreign Intervention', 'Foreign Diplomacy', 'Military Policy']]
    topics_df.loc[34] = [10, 'John Tyler', ['America', 'Foreign Intervention', 'Legislation', 'Internal Tension', 'Foreign Intervention', 'Foreign Intervention', 'Internal Tension', None, 'Foreign Intervention', 'Foreign Intervention']]
    topics_df.loc[35] = [9, 'William Harrison', ['America', None, None, None, None, None, None, None, None, None]]
    topics_df.loc[36] = [8, 'Martin Van Buren', ['America', ['Foreign Intervention, Crime'], ['Foreign Intervention, Crime'], 'Foreign Diplomacy', None, 'America', None, None, None, None]]
    topics_df.loc[37] = [7, 'Andrew Jackson', ['America', 'Economy', 'America', 'Foreign Intervention', 'Foreign Intervention', 'Legislation', None, None, 'Foreign Intervention', 'Internal Tension']]
    topics_df.loc[38] = [6, 'John Quincy Adams', ['America', ['America', 'Values'], 'Internal Tension', 'Foreign Intervention', 'America', None, None, None, None, None]]
    topics_df.loc[39] = [5, 'James Monroe', ['America', None, None, None, None, None, None, None, None, None]]
    topics_df.loc[40] = [4, 'James Madison', [['US War', 'America'], 'Faith', None, None, 'US War', ['Faith', 'Internal Tension'], 'Economy', None, 'Legislation', 'Foreign Intervention']]
    topics_df.loc[41] = [3, 'Thomas Jefferson', ['America', 'Values', None, 'Crime', None, ['Values', 'Faith'], 'Military Policy', 'Internal Tension', 'Foreign Diplomacy', None]]
    topics_df.loc[42] = [2, 'John Adams', ['America', 'Crime', 'Historical Reference', 'Faith' , None, None, None, None, None, None]]
    topics_df.loc[43] = [1, 'George Washington', ['America', 'Crime', 'Foreign Diplomacy', 'America', None, 'Values', 'Foreign Diplomacy', 'Foreign Diplomacy', 'Faith', None]]

    return topics_df

df = topics_table()

for idx, president in df['president'].iteritems():
    print(df)
    print(idx)
    print(df.loc[idx]['topics_list'])
    president_total_topics(president, df.loc[idx]['topics_list'])

