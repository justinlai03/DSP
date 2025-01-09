import pickle
import streamlit as st
import pandas as pd
import joblib
from transformers import pipeline
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import gdown, os

# Set page configuration
st.set_page_config(
    page_title="TikTok Misinformation Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar Navigation
sidebar = st.sidebar
sidebar.title("Navigation")
page = sidebar.radio("Select a page", ["Home", "Introduction", "Prediction", "Dashboard"])

# Load models and dataset
file_url = "https://drive.google.com/uc?id=1YpPVbcu4MEtdLi3l52K1CbpJWJbHxH6t"
output = 'emotion_analyzer-1.pkl'

if not os.path.exists('emotion_analyzer-1.pkl'):
    gdown.download(file_url, output, quiet=False)
else: 
    print(f"{output} already exists. Skipping download.")

with open(output, 'rb') as f:
    model = pickle.load(f)
misinformation_pipeline = joblib.load("misinformation_pipeline-v2.pkl")
# emotion_analyzer = joblib.load("C:/Users/user/OneDrive/Documents/UM/Y3S1/DSP/emotion_analyzer-1.pkl")
emotion_analyzer = model
data_path = "processed_data-v2.csv"

try:
    dataset = pd.read_csv(data_path)
except Exception as e:
    dataset = None
    dataset_error = str(e)

def load_image(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")
        
# Home Page
if page == "Home":
    logo = load_image("TruthTok_Logo-v2.png")  # Replace with the correct path if needed

    st.markdown(
    """
    <style>
        .flex-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            margin-top: 50px;
        }
        .flex-item-image {
            margin-bottom: 20px;
        }
        .center-text {
            text-align: center;
            margin-top: 20px;
        }
    </style>
    """, unsafe_allow_html=True
)

    # Render content with image above title
    st.markdown(
        f"""
        <div class="flex-container">
            <div class="flex-item-image">
                <img src="data:image/png;base64,{logo}" alt="TruthTok Logo">
            </div>
            <div class="flex-item-content">
                <h1 style="margin: 0;">Welcome to TruthTok!</h1>
            </div>
        </div>
        <div class="center-text">
            <p>Use this app to predict misinformation, analyze emotions, and explore trends.</p>
        </div>
        """, unsafe_allow_html=True
    )

# Introduction Page
elif page == "Introduction":
    st.header("Introduction")
    st.write(
        """
        With over 1.6 billion active users in 2024, TikTok has become a dominant platform for information consumption, 
        particularly among younger audiences who often rely on it for news and real-time updates. However, its algorithm-driven 
        recommendations and high-engagement features create an environment where misinformation can spread rapidly, often surfacing 
        in trending topics and search results, embedding false narratives into public discourse.

        Addressing these challenges requires innovative tools that combine real-time misinformation detection with sentiment and 
        emotional analysis to enable users to assess both the credibility and the emotional impact of TikTok content, fostering 
        a more informed and critical user base.
        """
    )

    st.subheader("Objectives")
    st.write(
        """
        1. Develop a prediction tool for users to input TikTok video data and receive instant predictions on misinformation.  
        2. Integrate sentiment and emotion analysis in the tool, allowing users to assess the emotional tone of videos and 
           understand its impact on engagement and misinformation spread.  
        3. Design a dashboard to visualise trends in misinformation, sentiment, emotions, and engagement, allowing users to filter 
           and explore key patterns.
        """
    )

    st.subheader("Dataset")
    if dataset is not None:
        st.write("Below is a preview of the dataset used for this project:")
        st.dataframe(dataset.head(10))
        st.write(f"Dataset Shape: {dataset.shape}")
    else:
        st.error("Failed to load dataset.")
        st.write(f"Error Details: {dataset_error}")

    st.subheader("Heatmap of Feature Correlations")
    emotion_data = pd.read_csv("sentiment_emotion_analysis_results-v2.csv")
    misinformation_data = pd.read_csv("data_with_text_length.csv")

    misinformation_data['claim_status'] = misinformation_data['claim_status'].replace({'opinion': 0, 'claim': 1})
    numerical_df = misinformation_data.select_dtypes(include=['float64', 'int64'])
    correlation_matrix = numerical_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
    ax.set_title('Heatmap of Feature Correlations')
    st.pyplot(fig)

    st.subheader("Emotion Distribution for Misinformation vs Truth")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.countplot(data=emotion_data, x='emotion', hue='claim_status', palette="muted", ax=ax2)
    ax2.set_title("Emotion Distribution for Misinformation vs Truth")
    st.pyplot(fig2)

# Prediction Page
elif page == "Prediction":
    st.header("Misinformation Prediction and Sentiment/Emotion Analysis")

    st.subheader("Enter Video Details")
    video_viewcount = st.number_input("Video View Count", min_value=0, step=1)
    like_count = st.number_input("Like Count", min_value=0, step=1)
    video_text = st.text_area("Video Transcription Text (max 500 words)", max_chars=3000)
    share_count = st.number_input("Share Count", min_value=0, step=1)
    download_count = st.number_input("Download Count", min_value=0, step=1)
    comment_count = st.number_input("Comment Count", min_value=0, step=1)

    input = pd.DataFrame({
        'video_transcription_text': [video_text],
        'video_view_count': [video_viewcount],
        'video_like_count': [like_count], 
        'video_share_count': [share_count], 
        'video_download_count': [download_count],
        'video_comment_count': [comment_count],
    })

    if st.button("Analyze and Predict"):
        if video_text:
            prediction = misinformation_pipeline.predict(input)[0]
            prediction_proba = misinformation_pipeline.predict_proba(input)[0]
            sentiment = TextBlob(video_text).sentiment.polarity
            sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
            emotion_result = emotion_analyzer(video_text)[0]
            emotion_label = emotion_result['label']

            st.write(f"Prediction: **{'Claim' if prediction == 1 else 'Opinion'}**")
            st.write(f"Confidence: **{prediction_proba[1] * 100:.2f}% Claim, {prediction_proba[0] * 100:.2f}% Opinion**")
            st.write(f"Sentiment: **{sentiment_label}** (Score: {sentiment:.2f})")
            st.write(f"Emotion: **{emotion_label}**")
        else:
            st.warning("Please enter the video transcription text.")

# Dashboard Page
elif page == "Dashboard":
    st.header("Misinformation Trend Dashboard")
    power_bi_url = "https://app.powerbi.com/view?r=eyJrIjoiYmRhY2E0NGItY2RhNy00ODMxLTg1M2MtNzQxNWFmNjRhOWY0IiwidCI6ImE2M2JiMWE5LTQ4YzItNDQ4Yi04NjkzLTMzMTdiMDBjYTdmYiIsImMiOjEwfQ%3D%3D"
    st.components.v1.iframe(src=power_bi_url, width=1000, height=625, scrolling=True)
