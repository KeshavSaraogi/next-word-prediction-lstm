import nltk
from nltk.corpus import gutenberg
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers.legacy import Adam

nltk.download('gutenberg', quiet=True)

text = gutenberg.raw('shakespeare-hamlet.txt').lower()

# Add a padding token to the text
text = "<PAD> " + text

max_vocab_size = 5000
tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<UNK>")
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

max_sequence_length = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = to_categorical(y, num_classes=total_words)

model = Sequential()
model.add(Embedding(total_words, 64, input_length=max_sequence_length - 1))
model.add(GRU(100))
model.add(Dropout(0.2))
model.add(Dense(total_words, activation='softmax'))

optimizer = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

history = model.fit(X, y, epochs=10, verbose=1)

model.save('next_word_model.h5')
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('training_accuracy.png')

plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_loss.png')

print("Model and tokenizer saved.")
print("Training history plots saved.")

# Test case
test_text = "to be"
test_sequence = tokenizer.texts_to_sequences([test_text])[0]
test_padded = pad_sequences([test_sequence], maxlen=max_sequence_length - 1, padding='pre')
predicted = model.predict(test_padded, verbose=0)
predicted_word_index = np.argmax(predicted)
predicted_word = tokenizer.index_word.get(predicted_word_index)

print("Test predicted word:", predicted_word)
print("Index word for 1:", tokenizer.index_word.get(1))
print("index word for 0:", tokenizer.index_word.get(0))