from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import time
import pandas as pd
import datetime

# Navigate to UVA Website
driver = webdriver.Chrome()
driver.get("https://millercenter.org/the-presidency/presidential-speeches")

# Make every speech acessible from 1 html page
current_len = 0
for i in range(90):
    print(i)
    time.sleep(2)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")



html = driver.page_source # Set html to the new page
soup = BeautifulSoup(html, 'html.parser') # Create BS_OBJ with full html
speeches = soup.findAll("div", {"class": "views-row"}) # Find every speech in the html
links = []
for obj in speeches: # Create list of links to every speech page
    speech_link = obj.find('a')
    links.append(speech_link)


speeches_df = pd.DataFrame(columns=['date', 'speech_title', 'president', 'text', 'paragraph', 'full_document'])
counter = 0
s_counter = 0
for link in links:
    print(str(s_counter + 1) + ' / ' + str(len(links)))
    
    text = []
    driver.get("https://millercenter.org/" + link['href'])
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    
    president = soup.find("p", {"class": "president-name"}).getText()
    title_split = link.contents[0].split(': ', 1)
    title = title_split[1]
    date = datetime.datetime.strptime(title_split[0].replace(',', ''), '%B %d %Y').strftime('%m/%d/%Y')
        
    try:
        text_container = soup.find("div", {"class": "expandable-text-container"})
        paragraphs = text_container.findAll("p")
        p_counter = 0
        p_length = len(paragraphs)
        for p in paragraphs:
            text = p.getText()
            if p_length == 1:
                full_document = 1
            else:
                full_document = 0
            speeches_df.loc[counter] = [date, title, president, text, p_counter, full_document]    
            p_counter += 1
            counter += 1
    except:
        try:
            text_container = soup.find("div", {"class": "view-transcript"})
            paragraphs = text_container.findAll("p")
            p_counter = 0
            p_length = len(paragraphs)
            for p in paragraphs:
                text = p.getText()
                if p_length == 1:
                    full_document = 1
                else:
                    full_document = 0
                speeches_df.loc[counter] = [date, title, president, text, p_counter, full_document]    
                p_counter += 1
                counter += 1
        except:
            text = 'error'
            counter += 1
            full_document = 2
            speeches_df.loc[counter] = [date, title, president, text, p_counter, full_document]
    s_counter += 1
    

speeches_df.to_csv('data/speeches_by_paragraph.csv', index='False')
driver.close()

#print(obj.find('a').contents)