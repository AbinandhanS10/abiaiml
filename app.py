import streamlit as st
import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Setup
st.set_page_config(page_title="Smart Email System", layout="wide")

# Download stopwords
nltk.download('stopwords')

# Load model
model = pickle.load(open("email_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

stop_words = set(stopwords.words('english'))

# Functions
def clean_text(text):
    text = re.sub(r'\W', ' ', str(text))
    text = text.lower().split()
    text = [word for word in text if word not in stop_words]
    return " ".join(text)

def predict_email(text):
    text = clean_text(text)
    vector = vectorizer.transform([text])
    return model.predict(vector)[0]

def get_reply(label):
    if label == 1:
        return "⚠️ Ignore this email"
    else:
        return "✅ Reply professionally"

# UI HEADER
st.title("📧 Smart Email Recommendation System")
st.markdown("### 🚀 AI-powered email classification and smart reply system")

# Sidebar
st.sidebar.header("⚙️ Options")
st.sidebar.write("Upload your CSV file to analyze emails")

# Upload
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    if 'text' not in data.columns:
        st.error("❌ CSV must contain a 'text' column")
    else:
        results = []

        for mail in data['text']:
            label = predict_email(mail)
            reply = get_reply(label)

            results.append({
                "Email": mail,
                "Prediction": "Spam" if label == 1 else "Normal",
                "Suggested Reply": reply
            })

        output_df = pd.DataFrame(results)

        # Metrics
        spam_count = (output_df["Prediction"] == "Spam").sum()
        normal_count = (output_df["Prediction"] == "Normal").sum()

        col1, col2 = st.columns(2)
        col1.metric("🚨 Spam Emails", spam_count)
        col2.metric("✅ Normal Emails", normal_count)

        st.divider()

        # Table
        st.subheader("📊 Results")
        st.dataframe(output_df, use_container_width=True)

        # Download button
        csv = output_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download Results",
            csv,
            "output_results.csv",
            "text/csv"
        )

# Footer
st.markdown("---")
st.markdown("💡 Developed as part of Smart Email Recommendation System Project")