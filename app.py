import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

with open('Email_spam_detection.ipynb', 'rb') as file:
    tfidf = pickle.load(file)

with open('Email_spam_detection (Model Building & Improvement).ipynb', 'rb') as file:
    model = pickle.load(file)

st.title("Email/SMS Spam classifier")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess

    transformed_sms = transform_text(input_sms)
    # 2.vectorize

    vector_input = tfidf.transform([transformed_sms])
    # 3.predict

    result = model.predict(vector_input)[0]
    # 4.Display

    if result==1:
        st.header("Spam")
    else:
        st.header("Not Spam")

