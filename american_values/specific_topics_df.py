import pandas as pd
from rhetoric.rhetoric_dataset import sql_upload

def topics_table():
    topics_df = pd.DataFrame(columns=['president_num','president', 'topic_1', 'topic_2', 'topic_3', 'topic_4', 'topic_5', 'topic_6', 'topic_7', 'topic_8', 'topic_9', 'topic_10', 'topic_11', 'topic_12', 'topic_13', 'topic_14', 'topic_15'])
    topics_df.loc[0] = [44, 'Donald Trump', 'American People', 'New America', 'Party Politics', 'Immigration', 'Parkland Teacher', 'Community', 'Fake News', 'Fossil Fuel Energy', 'Drug Addition', None, None, None, None, None, None]
    topics_df.loc[1] = [43, 'Barack Obama', 'American People', 'American Employment', 'Israeli/Palestinian Conflict', 'War On Terror', 'Israel', 'Gun Reform', 'Healthcare Reform', 'Equal Pay', 'Iraq War', 'Supreme Court Nomination', 'Cuba', None, None, None, None]
    topics_df.loc[2] = [42, 'George W. Bush', 'American People', 'American Values', 'Iraq War', 'Stem Cell Research', 'African AIDS Epidemic', 'Tax Cut', 'Hurricane Katrina', 'Auto Industry Failue', 'Illegal Immigration', 'Medicare Coverage', 'Community', 'Faith', None, None, None]
    topics_df.loc[3] = [41, 'Bill Clinton', 'American People', 'New America', 'Bosnian War', 'Oklahoma City Bombing', 'Somalian Affairs', 'Victims Rights', 'Northern Ireland Peace', 'Lewinsky Scandal', 'Ghana', 'Grand Forks Flood', None, None, None, None, None]
    topics_df.loc[4] = [40, 'George H. W. Bush', 'American People', 'New America', 'Gulf War', 'Soviet Peace Process', 'Somalian Affairs', 'Budget Deficit', 'Invasion of Panama', 'Disabilities Act', 'Soviet Peace Process', None, None, None, None, None, None]
    topics_df.loc[5] = [39, 'Ronald Reagan', 'American People', 'American Values', 'Soviet Peace Process', 'Israeli/Palestinian Conflict', 'Tax Cut', 'Labor Strike', 'Challenger Tragedy', 'Libya Air Strikes', 'Military Remembrance', 'Unclear Topic', 'Drug Abuse', None, None, None, None]
    topics_df.loc[6] = [38, 'Jimmy Carter', 'Energy Crisis', 'Cold War', None, 'Iran Hostage Crisis', 'Israeli/Palestinian Conflict', 'Chinese Relations', 'Panama Canal Treaty', 'Inflation', None, None, None, None, None, None, None]
    topics_df.loc[7] = [37, 'Gerald Ford', 'American People', 'Energy Crisis', 'Values', 'Soldier Amnesty', 'Watergate Scandal', 'New America', 'Pacific Islands', 'Global Cooperation', None, None, None, None, None, None, None]
    topics_df.loc[8] = [36, 'Richard Nixon', 'American People', 'New America', 'Vietnam War', 'Watergate Scandal', 'Watergate Scandal', 'Cold War', 'Internal Tension', 'Energy Embargo', 'Party Politics', None, None, None, None, None, None]
    topics_df.loc[9] = [35, 'Lyndon B. Johnson', 'American People', 'World Peace', 'Vietnam War', 'Dominican Intervention', 'Space Race', 'Nuclear Arms Treaty', 'Civil Rights', 'Faith', None, None, None, None, None, None, None]
    topics_df.loc[10] = [34, 'John F. Kennedy', 'New World', 'Womens Rights', 'Trade', 'Ole Miss Riot', 'News/Media', 'Cuban Missile Crisis', 'Berlin Freedom', None, 'Faith', 'Peace Corp Creation', None, None, None, None, None]
    topics_df.loc[11] = [33, 'Dwight D. Eisenhower', 'World Peace', 'Eisenhower Doctrine', None, 'Party Politics', 'Atomic Bomb', None, None, None, None, None, None, None, None, None, None]
    topics_df.loc[12] = [32, 'Harry S. Truman', 'World Peace', 'Party Politics', 'World War 2', 'Atomic Bomb', 'Labor Unions', 'Truman Doctrine', 'Korean War', None, None, None, None, None, None, None, None]
    topics_df.loc[13] = [31, 'Franklin D. Roosevelt', 'World War 2', 'American People', 'Public Works', 'World Peace', 'Banking Crisis', 'Supreme Court Legislation', 'Energy', 'Unemployment', None, None, None, None, None, None, None]
    topics_df.loc[14] = [30, 'Herbert Hoover', 'Economy', 'Treaties', 'Food Supply', 'Party Politics', 'Tariffs', 'Law Enforcement', 'Trade Exports', 'Farm Bill', 'World Peace', 'Foreign Debts', None, None, None, None, None]
    topics_df.loc[15] = [29, 'Calvin Coolidge', 'American Government','American People', 'Faith', 'George Washington', 'Mount Rushmore', None, 'Declaration of Independence', None, None, None, None, None, None, None, None]
    topics_df.loc[16] = [28, 'Warren G. Harding', 'American Government', 'Labor Conditions', 'Values', 'World War 1 Remembrance', 'Fair Labor Practices', 'Party Politics', None, 'Americanism', None, None, None, None, None, None, None]
    topics_df.loc[17] = [27, 'Woodrow Wilson', 'World Peace', 'World War 1 Germany', None, 'Declaration of Independence', 'American Values', 'World War 1', 'Values', 'Mexican Revolution', 'Monopolies', 'Womens Rights', None, None, None, None, None]
    topics_df.loc[18] = [26, 'William Taft', 'American Government', 'Trade Agreement With Canada', 'Tariffs', 'Income Tax', 'Environmental Protection', 'Tariff Legislation', 'Economic Legislation', None, None, None, None, None, None, None, None]
    topics_df.loc[19] = [25, 'Theodore Roosevelt', 'American Government', 'America', 'Panama Canal', 'Cuban Relations', None, 'Panama Relations', 'Puerto Rico', 'Meat Regulations', 'Racism', 'Neutrality', None, None, None, None, None]
    topics_df.loc[20] = [24, 'William McKinley', 'American People', 'American Government', 'Spanish American War', 'Government Spending', 'Spanish American War', 'Currency Report', None, None, None, None, None, None, None, None, None]
    topics_df.loc[21] = [25, 'Grover Cleveland', 'American Government', 'American People', 'Currency System', 'Neutrality', 'Chinese Exclusion Act', 'Racism', 'Party Politics', 'Civil Service Reform', 'Monroe Doctrine', 'Literacy Test Immigration Veto', 'Texas Seed Bill Veto', 'Statue of Liberty', None, None, None]
    topics_df.loc[22] = [22, 'Benjamin Harrison', 'American Government','American People', 'Internal Tension', 'Chilean Relations', 'Honoring President Garfield', 'Johnstown Flood', 'Treasury Secretary Nomination', 'Latin American Relations', 'Native American Relations', 'America', 'Faith', 'Historical Reference', None, None, None]
    topics_df.loc[23] = [21, 'Chester A. Arthur', 'American Government', 'Native American Relations', 'Public Safety Regulations Veto', 'Values', 'Historical Reference', 'Nicaragua Canal', 'Chinese Exclusion Act Veto', 'River and Harbors Act Veto', 'Central American Relations', None, None, None, None, None, None]
    topics_df.loc[24] = [20, 'James A. Garfield', 'American Government', None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    topics_df.loc[25] = [19, 'Rutherford B. Hayes', 'American Government', 'American People', 'Election Legislation', 'Military Legislation', 'Native American Relations', 'Banking System Veto', 'Party Politics', None, 'Coinage System Veto', 'Chinese Immigration Veto', None, 'Military Legislation Veto', None, None, None]
    topics_df.loc[26] = [18, 'Ulysses S. Grant', 'American Government', 'Internal Tension', 'Currency Legislation', None, 'Formation of Germany', 'Faith', 'Labor Regulation', 'Presidential Power', 'Congressmen Pay', None, None, None, None, None, None]
    topics_df.loc[27] = [17, 'Andrew Johnson', 'American Government', 'American People', 'Reconstruction', 'Party Politics', 'Civil War', 'Internal Tension', 'Law Enforcement', 'Death of Lincoln', None, None, None, None, None, None, None]
    topics_df.loc[28] = [16, 'Abraham Lincoln', 'American Government', 'American People', 'Slavery', None, 'Farewell', 'Civil War', 'Faith', 'Values', 'Henry Clay', 'Historical Reference', None, None, None, None, None]
    topics_df.loc[29] = [15, 'James Buchanan', 'American Government', 'American People', 'Party Politics', 'Slavery', 'Virginian Tensions', 'Domestic Occupation', 'US Expansion', None, None, 'Kansas', None, None, None, None, None]
    topics_df.loc[30] = [14, 'Franklin Pierce', 'American Government', 'American People', None, 'US Expansion', 'Law Enforcement', None, 'Army Health', 'Spanish Relations', 'US-Mexican Convention', 'Mexican Treaty', 'Establishment of Kansas', None, None, None, None]
    topics_df.loc[31] = [13, 'Millard Fillmore', 'American Government', 'Texas', 'Marshal Law',  None, None, None, None, None, None, None, None, None, None, None, None]
    topics_df.loc[32] = [12, 'Zachary Taylor', 'American Government','US Expansion', 'Foreign Diplomacy', None, None, None, None, None, None, None, None, None, None, None, None]
    topics_df.loc[33] = [11, 'James K. Polk', 'Mexican-American War', 'Missouri Compromise', 'Cherokee Relations', None, 'US Expansion', 'French Revolution', 'Army Size', None, None, None, None, None, None, None, None]
    topics_df.loc[34] = [10, 'John Tyler', 'American Government', 'Mexican-American Affairs', 'Economic Policy', 'Slavery', 'Native American Relations', 'US Expansion', 'Dorr Rebellion', None, 'Chinese Relations', 'Treaty With Texas', None, None, None, None, None]
    topics_df.loc[35] = [9, 'William Harrison', 'American Government', 'American People', None, None, None, None, None, None, None, None, None, None, None, None, None]
    topics_df.loc[36] = [8, 'Martin Van Buren', 'American Government', 'American Economy', 'Canadian Revolution', 'Law Enforcement', 'British Relations', None, None, None, None, None, None, None, None, None, None]
    topics_df.loc[37] = [7, 'Andrew Jackson', 'American Government', 'American Economy', 'Trade', 'American Government','Choctaw Relations', 'South Carolina', 'Spending Legislation Veto', None, None, 'Texas Independence', 'Party Politics', None, None, None, None]
    topics_df.loc[38] = [6, 'John Quincy Adams', 'American Government', 'American Values', 'Values', 'Amistad Africans', 'Native American Relations', None, None, None, None, None, None, None, None, None, None]
    topics_df.loc[39] = [5, 'James Monroe', 'American Government', None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    topics_df.loc[40] = [4, 'James Madison', 'War of 1812', 'America', 'Faith', None, None, 'Alexandria Protests', 'National Bank Veto', None, 'Internal Improvements Veto', 'Occupation of West Florida', None, None, None, None, None]
    topics_df.loc[41] = [3, 'Thomas Jefferson', 'American Government', 'American Citizens', 'Values', None, 'Law Enforcement', None, 'Faith', 'Military Policy', 'Internal Tension', 'Spanish Relations', None, None, None, None, None]
    topics_df.loc[42] = [2, 'John Adams', 'American Government', 'American Economy', 'Treason', 'Washingtons Death', 'Faith' , None, None, None, None, None, None, None, None, None, None]
    topics_df.loc[43] = [1, 'George Washington', 'American Government', 'American Citizens', 'Treason', 'Native American Relations', None, 'Values', 'Faith', None, None, None, None, None, None, None, None]

    return topics_df

df = topics_table()

sql_upload(df, 'rhetoric_capstone', 'specific_topics')

