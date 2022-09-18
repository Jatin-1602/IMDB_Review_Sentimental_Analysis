import io
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn as sk
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras import Sequential
from tensorflow.keras import layers


df = pd.read_csv("IMDB Dataset.csv")

data = df["review"]
labels = df["sentiment"]

data = data.to_numpy()
labels = labels.to_numpy()

train_data, test_data, train_label, test_label = train_test_split(data, labels,
                                                                  test_size=0.2,
                                                                  shuffle=True,
                                                                  random_state=52)

def encode_sentiment(sentiment):
    if sentiment == 'positive':
        return 1
    return 0

encode = np.vectorize(encode_sentiment)
train_label = encode(train_label)
test_label = encode(test_label)

# vocab_size = 0
# max_len = 0
# for item in train_data:
#     words = item.split()
#     vocab_size += len(np.unique(words))
#     max_len = max(max_len, len(words))


vocab_size = 112186
max_len = 2470
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
embedding_dim = 16

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_data)

tokenizer_json = tokenizer.to_json()
with io.open('tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))


word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_data)
train_padded = pad_sequences(train_sequences, 
                             maxlen=max_len,
                             padding=padding_type,
                             truncating=trunc_type)

test_sequences = tokenizer.texts_to_sequences(test_data)
test_padded = pad_sequences(test_sequences, 
                             maxlen=max_len,
                             padding=padding_type,
                             truncating=trunc_type)


model = Sequential([
    layers.Embedding(vocab_size, embedding_dim, input_length=max_len),
    layers.GlobalAveragePooling1D(),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

num_epochs = 40

history = model.fit(
    train_padded, train_label,
    batch_size=512,
    epochs=num_epochs,
    validation_data=(test_padded, test_label),
    shuffle=False,
)

model.save('imdb_model_112186_2470_512.h5')

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()
    plt.savefig(string + ".png")


plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

loss, accuracy = model.evaluate(test_padded, test_label)
