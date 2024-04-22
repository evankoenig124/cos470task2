import nltk
import math
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import re

# Download stopwords if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Function to calculate TF
def compute_tf(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove numbers and words containing "subject"
    tokens = [token for token in tokens if not (token.isdigit() or 'subject' in token.lower())]
    # Count word occurrences
    word_counts = Counter(tokens)
    # Compute term frequency (TF)
    tf_values = {word: count / len(tokens) for word, count in word_counts.items()}
    return tf_values

# Function to calculate IDF
def compute_idf(documents):
    # Count the number of documents containing each word
    document_count = Counter()
    for document in documents:
        tokens = set(word_tokenize(document))
        # Remove words containing "subject"
        tokens = [token for token in tokens if 'subject' not in token.lower()]
        document_count.update(tokens)
    # Compute inverse document frequency (IDF)
    num_documents = len(documents)
    idf_values = {word: math.log(num_documents / (document_count[word] + 1)) for word in document_count}
    return idf_values

# Function to calculate TF-IDF
def compute_tfidf(text, idf_values):
    tf_values = compute_tf(text)
    # Compute TF-IDF
    tfidf_values = {word: tf_values[word] * idf_values[word] for word in tf_values}
    return tfidf_values

# Function to process a text file and compute TF-IDF for each unigram
def process_text_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        # Read the content of the file
        text = file.read()
        # Split the text into individual documents (assuming each line is a document)
        documents = text.split('\n')
        
        # Compute IDF for all documents
        idf_values = compute_idf(documents)
        
        # Compute TF-IDF for each unigram in each document
        tfidf_results = {}
        for document in documents:
            tfidf_values = compute_tfidf(document, idf_values)
            tfidf_results.update(tfidf_values)
        
        # Sort the TF-IDF dictionary by values in descending order
        sorted_tfidf = dict(sorted(tfidf_results.items(), key=lambda item: item[1], reverse=True))
        
        # Select the top 100 TF-IDF values
        top_100_tfidf = {k: sorted_tfidf[k] for k in list(sorted_tfidf)[:100]}
            
        return top_100_tfidf

# Example usage
def tfidfrun(path):
    tfidf_dictionary = process_text_file(path)
    return tfidf_dictionary
