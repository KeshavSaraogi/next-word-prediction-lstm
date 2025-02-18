import streamlit                                as st
import numpy                                    as np
from tensorflow.keras.models                    import load_model
from tensorflow.keras.preprocessing.sequence    import pad_sequences
import pickle


# Load the model and the tokenizer
model = load_model('prediction_lstm.h5')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    
# Predict the next word
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

# Streamlit Applicaiton
st.title('Next Word Prediction with LSTM')
inputText = st.text_input("Enter The Sequence of Words", "To Be Or Not To")
if st.button("Predict Next Word"):
    sequenceLength = model.input_shape[1] + 1
    nextWord = predictNextWord(model, tokenizer, inputText, sequenceLength)
    st.write(f'Next Word: {nextWord}')
