import tensorflow as tf
from tensorflow.keras.models import load_model
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("Model_112186_2470_512/imdb_model_112186_2470_512.h5")


def predict(review):
    with open('tokenizer.json') as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)

    max_len = 2470
    trunc_type = 'post'
    padding_type = 'post'

    sequence = tokenizer.texts_to_sequences(review)
    padded = pad_sequences(sequence,
                           maxlen=max_len,
                           padding=padding_type,
                           truncating=trunc_type)

    result = model.predict(padded)
    for value in result:
        if value < 0.5:
            print("Negative")
        else:
            print("Positive")


reviews_list=["It's a nice movie",
              "Worst movie"
            ]

predict(reviews_list)