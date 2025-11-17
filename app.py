import streamlit as st
from openai import OpenAI
import pandas as pd
import plotly.express as px 

st.title("🥗 Customer Review Sentiment Analyzer")
st.markdown("Analyze the sentiment of customer reviews ☺️🥹😍😒🤢")

#OpenAI API Key Input
openai_api_key = st.sidebar.text_input(
    "Enter your OpenAI API Key:",
    type="password",
    help="You can get your API key from https://platform.openai.com/account/api-keys",)


def classify_sentiment_openai(review_text):
    client = OpenAI(api_key=openai_api_key)
    prompt = f'''
        Classify the following customer review. 
        State your answer
        as a single word, "positive", 
        "negative" or "neutral":    

        {review_text}
        '''
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt
            }
        ]
    ) 
    return completion.choices[0].message.content

#CSV fileuploaded
uploaded_file = st.file_uploader(
    "Choose a CSV file with customer reviews", 
    type=["csv"])

#Once the user uploads a CSV file, read and display it
if uploaded_file is not None:
    # Read the file 
    reviews_df = pd.read_csv(uploaded_file)

    #check if the data has a text column
    text_columns = reviews_df.select_dtypes(include=['object']).columns

    if len(text_columns) == 0:
        st.error("No text columns found in the uploaded CSV file.")

    #show a dropdown menu to select the text columns
    review_column = st.selectbox(
        "Select the column containing customer reviews:",
        text_columns
    )

    #analyze sentiment for each review
    reviews_df["sentiment"] = reviews_df[review_column].apply(classify_sentiment_openai)

    #display the sentiment distribution in metrics in 3 colums: Positive, Negative, Neutral
    #make the strings in the sentiment column title case
    reviews_df["sentiment"] = reviews_df["sentiment"].str.title()
    sentiment_counts = reviews_df["sentiment"].value_counts()   
    st.write(reviews_df)
    st.write(sentiment_counts)

    #Create 3 columns to display 3 metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        #show the number of positive reviews
        positive_count = sentiment_counts.get("Positive", 0)
        st.metric("Positive", positive_count,
                  f"{positive_count / len(reviews_df) * 100:.2f}%")
    with col2:
        #show the number of neutral reviews
        neutral_count = sentiment_counts.get("Neutral", 0)
        st.metric("Neutral", neutral_count,
                  f"{neutral_count / len(reviews_df) * 100:.2f}%")
    with col3:
        #show the number of negative reviews
        negative_count = sentiment_counts.get("Negative", 0)
        st.metric("Negative", negative_count,
                  f"{negative_count/ len(reviews_df) * 100:.2f}%")
        
    #display pie chart 
    import plotly.express as px
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Sentiment Distribution"
    )
    st.plotly_chart(fig)
    
