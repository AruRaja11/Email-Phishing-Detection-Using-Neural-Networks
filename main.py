import streamlit as st 
import pickle
import string
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import time

# Preprocessing function
def preprocess(text):
    table = str.maketrans("", "", string.punctuation)
    text = text.replace("-", "")
    text = text.split()
    text = [word.lower() for word in text]
    text = [word.translate(table) for word in text if word.isalpha() and len(word) > 1]
    text = " ".join(text)

    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    text = [lemmatizer.lemmatize(word, pos='v') for word in words]
    return " ".join(text)

# Prediction function
def predict_fraud(content, model, tokenizer):
    content = preprocess(content)
    sequences = tokenizer.texts_to_sequences([content])
    content_seq = pad_sequences(sequences, maxlen=300, padding="post")
    prediction = model.predict(content_seq)
    percentage = round((float(prediction[0][0]) * 100), 2)
    if prediction > 0.5:
        return 1, percentage
    else:
        return 0, percentage

# Streamlit App UI
st.set_page_config(page_title="Phishing Email Detector", page_icon="✉️", layout="centered")
st.title("✉️ Phishing Email Detection System")
st.markdown("### 🔍 Detect if an email is legitimate or a phishing attempt using Deep Learning.")

# Load Model and Tokenizer
with st.spinner("Loading model and tokenizer..."):
    model = load_model("model.h5")
    tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
    time.sleep(1)

# Input section
st.markdown("### 📝 Enter Email Details")
subject = st.text_input(label="Subject of the Email", placeholder="e.g. Urgent: Verify your account now!")
body = st.text_area(label="Body of the Email", placeholder="Enter the content of the email here...")

contents = subject + " " + body

# Prediction section
if st.button("🔎 Analyze Email"):
    if subject.strip() and body.strip():
        with st.spinner("Analyzing email..."):
            result, confidence = predict_fraud(contents, model, tokenizer)
            time.sleep(1.2)

        progress_placeholder = st.empty()
        for i in range(int(confidence)):
            progress_placeholder.progress(i + 1)
            time.sleep(0.01)

        st.subheader("📊 Prediction Result")
        if result == 1:
            st.error(f"🚨 The email seems **Phishing** with **{confidence}% confidence**.")
        else:
            st.success(f"✅ The email seems **Legitimate** with **{confidence}% confidence**.")

    else:
        st.warning("⚠️ Please enter both subject and body before analysis.")

# Model Summary section
with st.expander("🧠 Model Parameters"):
    total_params = model.count_params()
    trainable_params = sum([tf.size(v).numpy() for v in model.trainable_variables])
    non_trainable_params = total_params - trainable_params

    st.write("**📊 Model Parameter Details**")
    st.metric(label="Total Parameters", value=f"{total_params:,}")
    st.metric(label="Trainable Parameters", value=f"{trainable_params:,}")
    st.metric(label="Non-Trainable Parameters", value=f"{non_trainable_params:,}")

    # Optional: show parameter distribution in chart
    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Pie(
        labels=["Trainable", "Non-Trainable"],
        values=[trainable_params, non_trainable_params],
        hole=.5,
        textinfo='label+percent',
        marker=dict(colors=['#36BA98', '#F45B69'])
    )])
    fig.update_layout(title_text="Parameter Composition")
    st.plotly_chart(fig, use_container_width=True)