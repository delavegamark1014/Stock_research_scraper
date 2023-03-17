#setting up
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from transformers import pipeline
from bs4 import BeautifulSoup
import requests
import re
import csv

#setting up the model
model_name  = "human-centered-summarization/financial-summarization-pegasus"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

#setting up pipeline
monitored_tickers=['ETH']

#searching for new in Google and Yahoo Finance
print('searching for stock news for',monitored_tickers)
def search_news_links(ticker):
    search_url='https://www.google.com/search?q=yahoo+finance+{}&tbm=nws'.format(ticker)
    r=requests.get(search_url)
    soup=BeautifulSoup(r.text,'html.parser')
    atags=soup.find_all('a')
    hrefs=[link['href']for link in atags]
    return hrefs

raw_urls={ticker:search_news_links(ticker)for ticker in monitored_tickers}

#strip out unwanted URLs
print('cleaning urls.')
exclude_list=['maps','policies','preferences','accounts','support']
def strip_unwanted_urls(urls,exclude_list):
    val=[]
    for url in urls:
        if 'https://' in url and not any(exc in url for exc in exclude_list):
            res=re.findall(r'(https?://\S+)', url)[0].split('&')[0]
            val.append(res)
    return list(set(val))

cleaned_urls = {ticker:strip_unwanted_urls(raw_urls[ticker],exclude_list) for ticker in monitored_tickers}

#search and scrape cleaned urls
print('scraping new links.')
def scrape_and_process(URLs):
        ARTICLES=[]
        for url in URLs:
            r=requests.get(url)
            soup=BeautifulSoup(r.text,'html.parser')
            results=soup.find_all('p')
            text=[res.text for res in results]
            words=' '.join(text).split(' ')[:350]
            ARTICLE= ' '.join(words)
            ARTICLES.append(ARTICLE)
        return ARTICLES
articles={ticker:scrape_and_process(cleaned_urls[ticker])for ticker in monitored_tickers}

#summarize all articles
print('summarizing articles.')
def summarize(articles):
    summaries=[]
    for article in articles:
        input_ids=tokenizer.encode(article,return_tensors='pt')
        output=model.generate(input_ids,max_length=55,num_beams=5,early_stopping=True)
        summary=tokenizer.decode(output[0],skip_special_tokens=True)
        summaries.append(summary)
    return summaries

summaries = {ticker:summarize(articles[ticker]) for ticker in monitored_tickers}  

#add sentiment analysis
print('calculating sentiment')
sentiment=pipeline('sentiment-analysis')
scores={ticker:sentiment(summaries[ticker])for ticker in monitored_tickers}  

#exporting results
print('exporting results.')
def create_output(summaries,scores,urls):
    output=[]
    for ticker in monitored_tickers:
        for counter in range(len(summaries[ticker])):
            output_this = [
                            ticker,
                            summaries[ticker][counter],
                            scores[ticker][counter]['label'],
                            scores[ticker][counter]['score'],
                            urls[ticker][counter]
                          ]        
    output.append(output_this)
    return output
final_output=create_output(summaries,scores,cleaned_urls)
final_output.insert(0,['Ticker','Summary','Sentiment','Sentiment Score','URL'])

with open('summaries.csv',mode='w',newline='') as f:
    csv_writer=csv.writer(f,delimiter=',', quotechar="",quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerows(final_output)