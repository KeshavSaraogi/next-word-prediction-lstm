import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

st.set_page_config(page_title="Next Word Prediction with LSTM RNN")

@st.cache_resource
def load_resources():
    model = load_model('prediction_lstm.keras')
    with open('tokenizerLSTM.pickle', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_resources()

st.title("Next Word Prediction")

input_text = st.text_area("Enter your text:")
max_sequence_len = model.input_shape[1]

def predict_top_n_words(input_text, model, tokenizer, max_sequence_len, n=5):
    """Predicts the top n words and their probabilities."""
    sequence = tokenizer.texts_to_sequences([input_text])[0]
    sequence = pad_sequences([sequence], maxlen=max_sequence_len)
    predicted = model.predict(sequence, verbose=0)[0]
    top_n_indices = np.argsort(predicted)[-n:][::-1]
    top_n_words = []
    top_n_probabilities = []
    for index in top_n_indices:
        for word, idx in tokenizer.word_index.items():
            if idx == index:
                top_n_words.append(word)
                top_n_probabilities.append(predicted[index])
                break
    return top_n_words, top_n_probabilities

if st.button("Predict"):
    if input_text:
        top_words, probabilities = predict_top_n_words(input_text, model, tokenizer, max_sequence_len)

        st.subheader("Top 5 Predicted Words:")
        df = pd.DataFrame({'Word': top_words, 'Probability': probabilities})
        st.dataframe(df)

        st.subheader("Probability Visualization")
        st.bar_chart(df.set_index('Word'))

    else:
        st.warning("Please enter some text.")

if st.checkbox("Show Model Summary"):
    import io
    from contextlib import redirect_stdout
    with io.StringIO() as buf, redirect_stdout(buf):
        model.summary()
        summary = buf.getvalue()
    st.text(summary)

if st.checkbox("Show Tokenizer Information"):
    st.write(f"Vocabulary size: {len(tokenizer.word_index)}")
    st.write(f"First 10 words in index: {dict(list(tokenizer.word_index.items())[:10])}")

# --- Instructions ---
st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Enter some text in the text area.
2. Click the "Predict" button to see the top 5 predicted words and their probabilities.
3. Check the "Show Model Summary" or "Show Tokenizer Information" boxes for more details.
""")