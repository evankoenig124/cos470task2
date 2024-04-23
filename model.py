from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
import torch
import torch.nn as nn
from torch.optim import Adam
from tfidf import tfidfrun
import matplotlib.pyplot as plt

#Takes line of training text and cleans noise such as usernames or websites
def preprocess(text, max_length=512):

    new_text = []

    for t in text.split(" "):

        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        t = '' if t.isdigit() else t
        new_text.append(t)

    truncated_text = " ".join(new_text)[:max_length]

    return truncated_text

#Instantiates model, this case being roberta fine-tuned on twitter data
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

#Instantiate training loop
optimizer = Adam(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

#Loads training texts
def load_train_data(filepath):
    train_texts = ''
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            train_texts += line
    return train_texts

#Instantiates training text load function
anorexiadata= load_train_data('/Users/evankoenig/Documents/GitHub/cos470/anorexiadata.txt')
normaldata= load_train_data("/Users/evankoenig/Documents/GitHub/cos470/normaldata.txt")

#Gets labels and texts
train_texts = [anorexiadata, normaldata]
train_labels = [1, 0]

#Trains model using epoch number parameter
def train_model(num_epochs):
    avg_losses = []  # List to store average losses per epoch
    for epoch in range(num_epochs):
        total_loss = 0
        for text, label in zip(train_texts, train_labels):

            text = preprocess(text)

            encoded_input = tokenizer(text, return_tensors='pt')

            output = model(**encoded_input)
            logits = output.logits

            # Find loss
            loss = criterion(logits, torch.tensor([label]))

            # Training step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Calculate average loss and append to list
        avg_loss = total_loss / len(train_texts)
        avg_losses.append(avg_loss)

        # Print epoch and average loss
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss}")

    # Plot the average loss over epochs
    plt.plot(range(1, num_epochs+1), avg_losses, label='Average Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()

#test on text passed in
def test_text(filepath, tfidf_dict, true_sentiment, alpha=0.55, beta=0.45):
    counter = 0
    with open(filepath, 'r', encoding='utf-8') as file:
        for text in file:
            newtext = preprocess(text)
            
            # Tokenize input
            encoded_input = tokenizer(newtext, return_tensors='pt')

            # Forward pass
            output = model(**encoded_input)
            logits = output.logits

            # Predict sentiment
            probabilities = softmax(logits.detach().numpy(), axis=1)
            sentiment_label = np.argmax(probabilities)

            # Incorporate TF-IDF score
            text_tokens = newtext.split()
            tfidf_score = sum(tfidf_dict.get(word, 0) for word in text_tokens)

            # Combine sentiment prediction and TF-IDF score
            combined_score = (sentiment_label * alpha) + (tfidf_score * beta)

            # Convert combined score to binary sentiment label
            if combined_score > 0.5:
                combined_sentiment = 1
            else:
                combined_sentiment = 0

            # Check if combined sentiment matches true sentiment
            if true_sentiment == combined_sentiment:
                counter += 1

            print(f"T: {true_sentiment} P: {combined_sentiment} || {text}")

        print(counter)


# Train the model
train_model(num_epochs=5)

# Test a piece of text
#text_to_test = "i am so sad, i want to die. i hate my life."
test_text('/Users/evankoenig/Documents/GitHub/cos470/anorexia_labels.txt', tfidfrun('/Users/evankoenig/Documents/GitHub/cos470/anorexiadata.txt'),  0)
test_text('/Users/evankoenig/Documents/GitHub/cos470/normal_labels.txt', tfidfrun('/Users/evankoenig/Documents/GitHub/cos470/normaldata.txt'), 1)
