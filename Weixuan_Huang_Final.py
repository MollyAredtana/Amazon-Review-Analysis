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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

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


def Sentiment_Analysis(df):
    # This function will use TextBlob to find the sentiment polarity of the adjectives associated with each feature
    df['sentiment'] = df['content'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)



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
    # df['sentiment'] = df['content'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
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

def model_predict(df_copy):
    X = df_copy[['sentiment', 'Positive_Count', 'Negative_Count']]  # Features
    y = df_copy['Actual_Label']  # Target label

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% training and 30% testing

    # Create a logistic regression model
    model = LogisticRegression()  

    # Train the model
    model.fit(X_train, y_train)  

    # Predict on the test set
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Generate a confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{conf_matrix}")

    tp = conf_matrix[1, 1]
    tn = conf_matrix[0, 0]
    fp = conf_matrix[0, 1]
    fn = conf_matrix[1, 0]

    # Calculate True Positive Rate (TPR) and False Positive Rate (FPR)
    tpr = tp / (tp + fn)
    fpr = fp / (tn + fp)

    print(f"True Positive Rate (TPR): {tpr}")
    print(f"False Positive Rate (FPR): {fpr}")

    # Print classification report for additional details
    print(classification_report(y_test, y_pred))



def get_adjacent_words(df, target_words):
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

    return results

def extract_adjacent_words(text, target_words):
    # Check if the text is not a string
    if not isinstance(text, str):
        return {}  # Return an empty dictionary if the text is not a string

    # Tokenize the text and get POS tags
    tokens = nltk.word_tokenize(text)
    tagged_tokens = nltk.pos_tag(tokens)
    
    # Define POS tags
    noun_tags = {'NN', 'NNS', 'NNP', 'NNPS'}
    adjective_tags = {'JJ', 'JJR', 'JJS'}
    
    # Prepare to collect adjacent words
    results = {word: [] for word in target_words if word in tokens}
    
    # Look for each target word in the text
    for i, (token, tag) in enumerate(tagged_tokens):
        if token in target_words:
            start = max(0, i-2)
            end = min(len(tagged_tokens), i+3)
            
            # Collect adjacent words based on the tag of the target word
            if tag in noun_tags:
                # If the target word is a noun, look for adjacent adjectives
                for adj in tagged_tokens[start:end]:
                    if adj[1] in adjective_tags and adj[0] != token:
                        results[token].append(adj[0].lower())
            elif tag in adjective_tags:
                # If the target word is an adjective, look for adjacent nouns
                for noun in tagged_tokens[start:end]:
                    if noun[1] in noun_tags and noun[0] != token:
                        results[token].append(noun[0].lower())
                    
    return results

def apply_adjacent_extraction(row, target_words):
    adjacent_words = extract_adjacent_words(row['content'], target_words)
    # Flatten the list of adjacent words from all target words and join into a single string
    all_adjacent_words = [word for words_list in adjacent_words.values() for word in words_list]
    return ', '.join(all_adjacent_words)


def get_top_words(descriptive_words_dictionary, top_n):
    # Sort the dictionary by frequency in descending order and get the top 'n' entries
    sorted_words = sorted(descriptive_words_dictionary.items(), key=lambda item: item[1], reverse=True)
    top_words = [word for word, _ in sorted_words[:top_n]]

    return top_words

def find_adj(descriptive_dict):
    adj = []
    adjective_tags = {'JJ', 'JJR', 'JJS'}

    return adj

def txt_list(input_file):
    words_list = []

    # Open the text file and read the contents
    with open(input_file, 'r') as file:
        # Iterate through each line in the file
        for line in file:
            # Strip whitespace and newlines, then add the word to the list
            words_list.append(line.strip())

    # Now words_list contains all the words from the text file
    return words_list

def count_pos_neg_words(adjacent_words, positive_words, negative_words):
    # Split the string of words into a list
    words_list = adjacent_words.split(', ')
    # Count positive and negative words
    pos_count = sum(word in positive_words for word in words_list)
    neg_count = sum(word in negative_words for word in words_list)
    return pos_count, neg_count

if __name__ == '__main__':
    print("Please input 5 URLs from 5-star reviews to 1-star reviews in order.")
    urls = []
    all_data = []

    # Input URLs for different star ratings
    for i in range(5, 0, -1):
        url = input(f"Enter the URL for {i}-star reviews: ")
        urls.append(url)
    
    for url in urls:
        current_page = 0
        data = []
        print(f"Scraping reviews from: {url}")
        df = scrape(url, current_page, data)
        # print(df)
        all_data.append(df)  # Append data from each URL

    # Combine all data into one DataFrame
    df = pd.concat(all_data, ignore_index=True)

    
    # menu(token)
    # df = pd.read_csv("product.csv")
    # Clean, Prepare, and Convert data types
    # df['date'] = pd.to_datetime(df['date'], errors='coerce', format="%Y-%m-%d")  # Adjust the format as needed
    df = df.sort_values(by='date', ascending=False)

    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df['verified'] = df['verified'].apply(lambda x: 'Verified Purchase' if 'Verified Purchase' in str(x) else 'Not Verified')
    df['Actual_Label'] = (df['rating'] >= 4.0).astype(int)

    df.to_csv('product.csv', index=False)

    # # the sentiment function will create a new column to hold each review's sentiment score
    Sentiment_Analysis(df)

    # now we need load positive word and negative words
    # cited from https://gist.github.com/mkulakowski2/4289437
    # cited from https://gist.github.com/mkulakowski2/4289441
    positive_words = txt_list('positive-words.txt')
    negative_words = txt_list('negative-words.txt')
    # Perform EDA


    # perform actions here
    token, descriptive_words_dictionary = Get_Token(df)
    # a back up dictionary here
    descriptive_words_dictionary = dict(sorted(descriptive_words_dictionary.items(), key=lambda x: x[1], reverse=True))
    
    target_words = get_top_words(descriptive_words_dictionary, top_n=15)

    # get adject words for overall target words
    adjacent_words = get_adjacent_words(df, target_words)

    # get adject words for each review based on target word
    df['Adjacent_Words'] = df.apply(lambda row: apply_adjacent_extraction(row, target_words), axis=1)

    # now we will start predicting the model
    # make a copy first
    df_copy = df.copy()
    # Apply the function to each row in df_copy to create new columns
    df_copy['Positive_Count'], df_copy['Negative_Count'] = zip(*df_copy['Adjacent_Words'].apply(lambda x: count_pos_neg_words(x, positive_words, negative_words)))
    df_copy.to_csv("df_copy.csv") # for checking


    while True:
        option = input(
            "Please choose an option:\n"
            "1: Word Cloud Visualization\n"
            "2: General plots for the products\n"
            "3: Predict the Model\n"
            "4: Exit\n"
            "Enter your choice: ")
        
        if int(option) == 1:
            Word_Cloud_Visualization(adjacent_words) 
        
        elif int(option) == 2:
            plot_rating_distribution(df_copy)
            plot_sentiment_distribution(df_copy)
            compare_verified_reviews(df_copy)

        elif int(option) == 3:
            model_predict(df_copy)
        
        elif int(option) == 4:
            print("Program Ended")
            break
        else:
            print("Please Re-Enter the correct option")

    # print(Sentiment_Analysis(adjacent_words))



    

