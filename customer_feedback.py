import streamlit as st
import pandas as pd
import pickle
import re

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Customer Feedback Sentiment Analysis",
    layout="centered"
)

st.title("📊 Customer Feedback Sentiment Dashboard")

# ============================================
# LOAD MODEL
# ============================================

model = pickle.load(open("sentiment_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# ============================================
# LOAD DATASET
# ============================================

df = pd.read_csv("Womens Clothing E-Commerce Reviews.csv")
df = df[['Review Text', 'Rating']]
df.dropna(inplace=True)

df['Sentiment'] = df['Rating'].apply(lambda x: "Positive" if x > 3 else "Negative")

# ============================================
# TEXT CLEANING
# ============================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

# ============================================
# SIDEBAR MENU
# ============================================

menu = st.sidebar.radio(
    "Menu",
    ["Dataset", "Visualization", "Predict Sentiment"]
)

# ============================================
# DATASET PAGE
# ============================================

if menu == "Dataset":
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20))
    st.write("Shape:", df.shape)

# ============================================
# VISUALIZATION PAGE
# ============================================

elif menu == "Visualization":

    st.subheader("Sentiment Distribution")

    fig, ax = plt.subplots()
    sns.countplot(x=df['Sentiment'], ax=ax)
    st.pyplot(fig)

    st.subheader("WordCloud (Positive Reviews)")

    text = " ".join(df[df['Sentiment']=="Positive"]['Review Text'])

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate(text)

    fig2, ax2 = plt.subplots(figsize=(10,5))
    ax2.imshow(wordcloud)
    ax2.axis("off")

    st.pyplot(fig2)

# ============================================
# PREDICTION PAGE
# ============================================

elif menu == "Predict Sentiment":

    st.subheader("Enter Review Text")

    user_input = st.text_area("Type here")

    if st.button("Predict"):

        cleaned = clean_text(user_input)
        vector = tfidf.transform([cleaned])
        prediction = model.predict(vector)

        if prediction[0] == 1:
            st.success("Positive Sentiment 😊")
        else:
            st.error("Negative Sentiment 😔")
