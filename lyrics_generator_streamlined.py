import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from helpers import *
import tensorflow as tf
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM, Bidirectional, Embedding
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.data import Dataset

def data_prep(text, max_len):
    # Tokenize.
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    total_words = len(tokenizer.word_index) + 1
    print('Total tokens: ', total_words)  # Not all tokens. Only ones that have some number of minimum appearances.

    # Create input sequences and targets.
    sequences = tokenizer.texts_to_sequences(text)
    sequences = [item for lst in sequences for item in lst]

    X = []
    y = []
    for i in range(len(sequences) - max_len): 
        end_slice = i + max_len + 1
        X.append(sequences[i: i + max_len])
        y.append(sequences[i + max_len])
    
    # Pad sequences for consistent input length
    padded_X = pad_sequences(X, maxlen=max_len, padding='pre')
    targets = np.array(y)

    return tokenizer, padded_X, targets, total_words


def create_model(num_words):
    print('Build model...')
    model = Sequential()
    model.add(Embedding(input_dim=num_words, output_dim=1024))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dense(num_words))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

    return model


def generate_text(model, tokenizer, max_sequence_length, seed_text, num_words):
    for _ in range(num_words):
        sequence = tokenizer.texts_to_sequences([seed_text])[0]
        padded_sequence = pad_sequences([sequence], maxlen=max_sequence_length, padding='pre')
        predicted_word_index = np.argmax(model.predict(padded_sequence), axis=-1)
        predicted_word = tokenizer.index_word[predicted_word_index[0]]
        seed_text += " " + predicted_word
    return seed_text


# PARAMS
NUM_ARTIST = 200 # Need to change this so that gets the actual number of artists of the genres. Currently just looks for `NUM_ARTIST` artists and 
# of those, gets the artist of the specific genre. That's why was getting error before.
MAX_SONGS = 10

# Process data into dataframe.
path1 = '/Users/randallpulido/Desktop/ML/lyrics_generator/artist_data/top_10000_artists/10000-MTV-Music-Artists-page-1.csv'
artist_genre_df = get_artists_by_genre(path1, num_artists=NUM_ARTIST)
lyrics_df = get_lyrics(artist_genre_df['artist'].tolist(), max_songs=MAX_SONGS)
merged = lyrics_df.merge(artist_genre_df, on='artist', how='left')

# Get lyric data.
text = split_lyric_data(merged['lyrics']) # can keep like this for now 
# but should optimize in future i.e. not use split_lyric_data function
# instead opt for combining `lyrics` entries of merged df.

MAX_SEQ = 5
BATCH_SIZE = 32
BUFFER_SIZE = 1000

tokenizer, X, y, num_words = data_prep(text, MAX_SEQ)

dataset = Dataset.from_tensor_slices((X, y))
dataset = dataset.shuffle(buffer_size=1000).batch(BATCH_SIZE)

model = create_model(num_words)

# Train the model
model.fit(X, y, epochs=20, verbose=2)

model.save('./streamlined_models/lyrics_generator_1', save_format='tf')

seed_text = 'For the love of'
generated_text = generate_text(model, tokenizer, MAX_SEQ, seed_text, 50)
print(generated_text)
