from sklearn.feature_extraction.text import TfidfVectorizer

# Example documents
documents = [
    test.ssv
]

# Initialize TfidfVectorizer with bigram tokenization
vectorizer = TfidfVectorizer(ngram_range=(2, 2))

# Fit and transform the documents
tfidf_matrix = vectorizer.fit_transform(documents)

# Get feature names (bigrams)
feature_names = vectorizer.get_feature_names_out()

# Print the TF-IDF matrix and feature names
print("TF-IDF Matrix (Bigrams):")
print(tfidf_matrix.toarray())
print("Feature Names (Bigrams):")
print(feature_names)
