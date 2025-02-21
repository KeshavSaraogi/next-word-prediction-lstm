import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import os

model_names = {
    "Many-to-One": ("next_word_model.h5", "tokenizer.pickle"),  # Only Many-to-One
}

st.title("Next Word Predictor")
st.write("Powered by GRU RNN")

model_selection = st.selectbox("Choose a model", list(model_names.keys()))

model_path, tokenizer_path = model_names[model_selection]

# Use absolute paths
model_path = os.path.abspath(model_path)
tokenizer_path = os.path.abspath(tokenizer_path)

# Load the model
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load the tokenizer
try:
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
except Exception as e:
    st.error(f"Error loading tokenizer: {e}")
    st.stop()

# Debugging: Print tokenizer information
print("Tokenizer word index length:", len(tokenizer.word_index))
print("Index word for 1:", tokenizer.index_word.get(1))
print("index word for 0:", tokenizer.index_word.get(0))

# Get the sequence length from the model's input shape
sequence_length = model.input_shape[1] + 1

def predict_next_sequence(model, tokenizer, text, sequence_length, num_words):
    token_list = tokenizer.texts_to_sequences([text])[0]
    padded_sequence = pad_sequences([token_list], maxlen=sequence_length - 1, padding='pre')
    predicted_sequence = model.predict(padded_sequence, verbose=0)[0]

    predicted_words = []
    for prediction in predicted_sequence:
        predicted_word_index = np.argmax(prediction)
        predicted_word = tokenizer.index_word.get(predicted_word_index)

        if predicted_word:
            predicted_words.append(predicted_word)
        else:
            predicted_words.append("<UNK>")

        if len(predicted_words) >= num_words:
            break

    return " ".join(predicted_words)

input_text = st.text_area("Enter text:")

if st.button("Generate Text"):
    if input_text:
        predicted_sequence = predict_next_sequence(model, tokenizer, input_text, sequence_length, 13)
        st.write("Input:", input_text)
        st.write("Predicted Sequence:", predicted_sequence)
    else:
        st.write("Please enter some text.")

st.write("Built with Streamlit and TensorFlow/Keras")

st.subheader("Model Info")
st.write(f"Selected Model: {model_selection}")

for i, layer in enumerate(model.layers):
    st.write(f"Layer {i}: {layer.name} ({layer.output_shape[-1]} dim)")