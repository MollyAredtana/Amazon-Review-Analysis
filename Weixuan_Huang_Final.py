import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import validators
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from collections import Counter
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from wordcloud import WordCloud

custom_headers = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Sec-Ch-Ua": "\"Google Chrome\";v=\"123\", \"Not:A-Brand\";v=\"8\", \"Chromium\";v=\"123\"",
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": "\"macOS\"",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "cross-site",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
}

def get_soup(url):
    response = requests.get(url, headers=custom_headers)
    if response.status_code != 200:
        print("Error in getting webpage")
        return None

    soup = BeautifulSoup(response.text, "lxml")
    return soup


def is_valid_url(url): 
    """Check if the provided URL is valid."""
    if validators.url(url):
        # print("The URL is valid.")
        return True
    else:
        print("The URL is not valid.")
        return False


def get_reviews(soup):
    review_elements = soup.select("div.review")
    scraped_reviews = []

    # Check if 'From other countries' header exists, stop scraping if found
    if soup.find("h3", string="From other countries"):
        return scraped_reviews

    for review in review_elements:
        r_author_element = review.select_one("span.a-profile-name")
        r_author = r_author_element.text if r_author_element else None

        r_rating_element = review.select_one("i.review-rating")
        r_rating = r_rating_element.text.replace("out of 5 stars", "") if r_rating_element else None

        r_title_element = review.select_one("a.review-title")
        r_title_span_element = r_title_element.select_one("span:not([class])") if r_title_element else None
        r_title = r_title_span_element.text if r_title_span_element else None

        r_content_element = review.select_one("span.review-text")
        r_content = r_content_element.text if r_content_element else None

        r_date_element = review.select_one("span.review-date")
        r_date = r_date_element.text if r_date_element else None

        r_verified_element = review.select_one("span.a-size-mini")
        r_verified = r_verified_element.text if r_verified_element else None

        r_image_element = review.select_one("img.review-image-tile")
        r_image = r_image_element.attrs["src"] if r_image_element else None

        r = {
            "author": r_author,
            "rating": r_rating,
            "title": r_title,
            "content": r_content,
            "date": r_date,
            "verified": r_verified,
            "image_url": r_image
        }

        scraped_reviews.append(r)

    return scraped_reviews

def update_url_for_next_page(url, current_page):
    # Parse the URL into components
    parsed_url = urlparse(url)
    # Extract query parameters into a dictionary
    query_params = parse_qs(parsed_url.query)
    
    # Update the 'pageNumber' parameter to the next page
    query_params['pageNumber'] = [current_page + 1]  # Increment page number
    
    # Re-encode the query parameters back into a query string
    new_query_string = urlencode(query_params, doseq=True)
    
    # Construct the new URL with the updated query string
    new_url = urlunparse(parsed_url._replace(query=new_query_string))
    
    return new_url

def get_product_name(soup):
    """
    Extracts the product's name from the Amazon product review page.

    :param soup: BeautifulSoup object of the Amazon product review page.
    :return: The product's name as a string.
    """
    # Assuming the product name is within an <h1> tag with a specific class or ID.
    # You'll need to replace 'h1.product-title' with the correct selector for the product's name.
    product_name_tag = soup.select_one('h1.product-title')  # Update this selector based on your inspection.
    
    if product_name_tag:
        return product_name_tag.get_text(strip=True)
    else:
        print("Product name not found.")
        return None

        
def is_captcha_or_blocked(soup):
    if soup.find("title").text == "Sorry! Something went wrong!":
        print("Might be blocked or faced with a CAPTCHA. Check the URL manually in a browser.")
        return True
    return False


def scrape(url, current_page, data):
    while True:
        if is_valid_url(url):
            soup = get_soup(url)
            reviews = get_reviews(soup)  # This should return a list of dictionaries
            if reviews:  # Check if reviews were found
                data.extend(reviews)  # Extend data with the new reviews

                current_page += 1
                url = update_url_for_next_page(url, current_page)
                # print("url is : ", url)
            else:  # No reviews found, might be the last page
                break
        
        else:
            break
    
    # Create DataFrame from the list of dictionaries
    df = pd.DataFrame(data)
    df.to_csv("product.csv")
    return df


def Get_Token(df):
    # content is the review
    token_list = []
    descriptive_token_dict = {}
    stop_words = set(stopwords.words('english'))
    words = set(nltk.corpus.words.words())
    descriptive_tags = set(['NN', 'NNP', 'NNPS', 'NNS', 'JJ', 'JJR', 'JJS'])

    for row in df['content']:
        # Skip rows with NaN values or non-string values
        if pd.isna(row) or not isinstance(row, str):
            continue
        tokens = nltk.word_tokenize(row)
        # Remove stopwords and non-English words
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.lower() in words]
        # POS tagging
        tagged_tokens = nltk.pos_tag(filtered_tokens)

        # We're interested in Nouns (NN, NNP, NNPS, NNS) and Adjectives (JJ, JJR, JJS)
        descriptive_words = [word for word, tag in tagged_tokens if tag in descriptive_tags]
        # Frequency analysis
        word_freq = Counter(descriptive_words)
        
        # Merge word frequency dictionary with the main dictionary
        for word, freq in word_freq.items():
            if word in descriptive_token_dict:
                descriptive_token_dict[word] += freq
            else:
                descriptive_token_dict[word] = freq


        # Display most common words
        # common_words = Counter(descriptive_token_dict).most_common()
        # print(common_words)
        token_list.append(filtered_tokens)
    


    return token_list, descriptive_token_dict


def Sentiment_Analysis(adjacent_words_dict):
    # This function will use TextBlob to find the sentiment polarity of the adjectives associated with each feature
    sentiment_results = {}
    
    for feature, adjectives in adjacent_words_dict.items():
        # Combine all adjectives into one text string for analysis
        adjective_text = ' '.join(adjectives)
        blob = TextBlob(adjective_text)
        # Calculate the sentiment polarity
        sentiment = blob.sentiment.polarity
        # Store the results
        sentiment_results[feature] = sentiment
    
    return sentiment_results


def Word_Cloud_Visualization(adjacent_words_dict):
    # Combine all the adjacent words into one text string for each feature
    text_per_feature = {feature: ' '.join(adj_words) for feature, adj_words in adjacent_words_dict.items()}

    # Generate a word cloud image for each feature
    wordclouds = {}
    for feature, text in text_per_feature.items():
        wordcloud = WordCloud(width = 800, height = 400, background_color ='white').generate(text)
        wordclouds[feature] = wordcloud

    # Display the generated image:
    for feature, wordcloud in wordclouds.items():
        plt.figure(figsize = (10, 5), facecolor = None)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"Word Cloud for '{feature}'")
        plt.tight_layout(pad = 0)
        plt.show()



# EDA Functions
def plot_rating_distribution(df):
    sns.countplot(x='rating', data=df)
    plt.title('Rating Distribution')
    plt.show()

def plot_sentiment_distribution(df):
    df['sentiment'] = df['content'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    sns.histplot(df['sentiment'], kde=True)
    plt.title('Sentiment Distribution')
    plt.show()


def compare_verified_reviews(df):
    verified_sentiment = df[df['verified'] == 'Verified Purchase']['sentiment']
    unverified_sentiment = df[df['verified'] != 'Verified Purchase']['sentiment']
    sns.histplot(verified_sentiment, color="green", kde=True, stat="density", linewidth=0)
    sns.histplot(unverified_sentiment, color="red", kde=True, stat="density", linewidth=0)
    plt.title('Sentiment Distribution for Verified vs. Non-verified Purchases')
    plt.show()

def menu(token, adj_words):
    df = pd.read_csv("product.csv")
    while True:
        print("Option 1 -- Features and Attributes: Analyze the text to determine which features of the product are most praised or criticized.")
        print("Option 2 -- Text Sentiment Analysis: Perform sentiment analysis on the review texts to determine overall sentiment and compare it with the given star ratings.")
        print("Option 3 -- Keyword Frequency and Word Clouds: Identify common themes by looking for frequently mentioned words or phrases in the review content.")
        print("Option 4 -- Price-Value Correlation: Consider analyzing text for mentions of the product's price or value for money and correlate this with ratings.")
        option = input("Please enter the option number: ")

        if int(option) not in [1,2,3,4]:
            print("Please re-enter the option number.")
        else:
            if int(option) == 1: # word cloud
                Word_Cloud_Visualization(adj_words)
            if int(option) == 2: # Text Sentiment Analysis
                Sentiment_Analysis(adj_words)
            # if int(option) == 3: # Keyword Frequency and Word Clouds
            #     Keyword_freq_Word_clouds(token)
            # if int(option) == 4: # Price-Value Correlation
            #     Price_Value_Corr(token)

def extract_adjacent_words(df, target_words):
    results = {word: [] for word in target_words}
    noun_tags = {'NN', 'NNS', 'NNP', 'NNPS'}
    adjective_tags = {'JJ', 'JJR', 'JJS'}

    for review in df['content'].dropna():
        tokens = nltk.word_tokenize(review)
        tagged_tokens = nltk.pos_tag(tokens)
        
        for i, (token, tag) in enumerate(tagged_tokens):
            if token in target_words:
                start = max(0, i-2)
                end = min(len(tagged_tokens), i+3)
                for adj_or_noun in tagged_tokens[start:end]:
                    if adj_or_noun[1] in noun_tags and token != adj_or_noun[0]:
                        results[token].append(adj_or_noun[0].lower())
                    elif adj_or_noun[1] in adjective_tags and token != adj_or_noun[0]:
                        results[token].append(adj_or_noun[0].lower())

    # # Debug: Print results to check what is being captured
    # for key, value in results.items():
    #     print(f"Word: {key}, Adjacent: {value}")

    return results

def get_top_words(descriptive_words_dictionary, top_n):
    # Sort the dictionary by frequency in descending order and get the top 'n' entries
    sorted_words = sorted(descriptive_words_dictionary.items(), key=lambda item: item[1], reverse=True)
    top_words = [word for word, _ in sorted_words[:top_n]]

    return top_words


if __name__ == '__main__':
    current_page = 0
    # # while True:
    # url = input("Please enter a valid URL: ")

    # data = []

    # df = scrape(url, current_page, data)

    # menu(token)
    df = pd.read_csv("product.csv")
    # Clean and convert data types
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df['verified'] = df['verified'].apply(lambda x: 'Verified Purchase' if 'Verified Purchase' in str(x) else 'Not Verified')
    df['date'] = pd.to_datetime(df['date'], errors='coerce', format="%Y-%m-%d")  # Adjust the format as needed

    token, descriptive_words_dictionary = Get_Token(df)
    # a back up dictionary here
    descriptive_words_dictionary = dict(sorted(descriptive_words_dictionary.items(), key=lambda x: x[1], reverse=True))
    target_words = get_top_words(descriptive_words_dictionary, top_n=5)

    # get adject words
    adjacent_words = extract_adjacent_words(df, target_words)


    # Perform EDA

    plot_rating_distribution(df)
    plot_sentiment_distribution(df)
    compare_verified_reviews(df)

    # Word_Cloud_Visualization(adjacent_words) 

    # print(Sentiment_Analysis(adjacent_words))


