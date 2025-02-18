import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the model and the tokenizer
try:
    lstmModel = load_model('prediction_lstm.h5')
    gruModel = load_model('prediction_lstm_gru.h5')

    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
except Exception as e:
    st.error(f"Error loading models or tokenizer: {e}")
    st.stop()  # Stop execution if models fail to load

def predictNextWord(model, tokenizer, text, sequence_length):
    tokenList = tokenizer.texts_to_sequences([text])[0]
    
    if len(tokenList) >= sequence_length:
        tokenList = tokenList[-(sequence_length - 1) : ]
    
    tokenList           = pad_sequences([tokenList], maxlen = sequence_length - 1, padding = 'pre')
    predictedWord       = model.predict(tokenList, verbose = 0)
    predictedWordIndex  = np.argmax(predictedWord, axis = 1)
    
    for word, index in tokenizer.word_index.items():
        if index == predictedWordIndex:
            return word
    return None

# Streamlit App
st.title('Next Word Prediction')

lstmInputText = st.text_area("Enter Input Sequence for LSTM", "To be or not to be, that is the")
gruInputText = st.text_area("Enter Input Sequence for GRU", "What a piece of work is man")

# Check for empty input
if not lstmInputText.strip() or not gruInputText.strip():
    st.warning("Please enter text for both models.")
    st.stop()

if st.button("Predict Next Word"):
    lstmSequenceLength = lstmModel.input_shape[1] + 1
    try:
        lstmNextWord = predictNextWord(lstmModel, tokenizer, lstmInputText, lstmSequenceLength)
        st.write(f'**LSTM Next Word:** {lstmNextWord}')
    except Exception as e:
        st.error(f"Error during LSTM prediction: {e}")

    gruSequenceLength = gruModel.input_shape[1] + 1
    try:
        gruNextWord = predictNextWord(gruModel, tokenizer, gruInputText, gruSequenceLength)
        st.write(f'**GRU Next Word:** {gruNextWord}')
    except Exception as e:
        st.error(f"Error during GRU prediction: {e}")
