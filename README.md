# Stock_scraper
This is a Python script that scrapes news articles related to a stock ticker symbol and summarizes them using the Pegasus model for financial summarization. The script then performs sentiment analysis on the summaries and exports the results into a CSV file.

## Getting Started
### Prerequisites
To run this project, you will need the following

* Python 3.x

* Transformers library

* BeautifulSoup library

* requests library

* csv library

You can install the required libraries using pip install:

### Installation:

Clone this repository:
```
git clone https://github.com/your-username/stock_scraper.git
```

Install the required libraries as mentioned above.
```
pip install transformers bs4 requests csv
```
### Executing program

* Run the stock_scraper.py file.
* Enter a stock ticker when prompted.
* The program will search for news articles about the selected stock, summarize them, analyze them for sentiment, and export the results to a CSV file named "summary.csv"
* Open the csv file for the results of the selected stock

### Contributing
Contributions are welcome! If you find a bug or want to add a feature, please submit an issue or a pull request.

### Acknowledgements
* This project uses the Pegasus financial summarization model from Hugging Face Transformers.
