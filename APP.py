import textblob
import pandas as pd
import streamlit as st
import cleantext
import matplotlib.pyplot as plt

# Display logo or 
st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRHUZ8YL1w09jUgWn3_WJXsCS0_Ol_Hl7v3TQ&s', width=200)

st.header('Sentiment Analysis')

# Analyze Text Section
with st.expander('Analyze Text'):
    text = st.text_input('Enter Text')
    if text:
        if st.button('Analyze'):
            blob = textblob.TextBlob(text)
            # Polarity thresholds
            if blob.sentiment.polarity >= 0.5:
                st.write('Positive Sentiment')
            elif blob.sentiment.polarity <= -0.5:
                st.write('Negative Sentiment')
            else:
                st.write('Neutral Sentiment')
            st.write(f"**Polarity Score:** {blob.sentiment.polarity}")
            st.info("Note: This analysis works best for English text.")

# Clean Text Section
pre = st.text_input('Clean Text')
if pre:
    if st.button('Clean Text'):
        st.write(cleantext.clean(pre))

# Analyze CSV Section
with st.expander('Analyze CSV'):
    file = st.file_uploader('Upload CSV File', type=['csv'])

    def score(x):
        blob = textblob.TextBlob(x)
        if blob.sentiment.polarity >= 0.5:
            return 'Positive Sentiment'
        elif blob.sentiment.polarity <= -0.5:
            return 'Negative Sentiment'
        else:
            return 'Neutral Sentiment'

    if file:
        try:
            # Read uploaded CSV file
            df = pd.read_csv(file)
            if 'text' not in df.columns:
                st.error("The uploaded CSV file must contain a 'text' column.")
            else:
                with st.spinner('Analyzing...'):
                    df['Sentiment'] = df['text'].apply(score)
                    st.write(df)

                    # Visualization
                    st.bar_chart(df['Sentiment'].value_counts())
                st.success('Analysis Completed!')
        except Exception as e:
            st.error(f"An error occurred: {e}")

# About Section
st.subheader('About')
st.write('This is a simple web app for sentiment analysis.')

st.subheader('How to use')
st.write('1. Enter text in the text box and click on the "Analyze" button.')
st.write('2. Upload a CSV file to analyze the sentiment of text in the "text" column.')
st.write('3. Use the "Clean Text" feature to clean your input text.')

# Developer Information
st.subheader('Developer Information')
st.write('Muhammad Ilyas')
st.subheader('Contact')
st.write('Email: [ilyaslayyah786@gmail.com](mailto:ilyaslayyah786@gmail.com)')
st.write('LinkedIn: [Muhammad Ilyas](https://www.linkedin.com/in/muhammad-ilyas-a59bb0289)')
