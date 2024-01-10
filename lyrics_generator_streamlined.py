from sklearn.model_selection import train_test_split
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
from tensorflow.keras.preprocessing.sequence import pad_sequences, TimeseriesGenerator
from tensorflow.data import Dataset
import json

def data_prep(songs, max_len):
    # Split data into train, validation, and test.
    train_songs, val_test = train_test_split(songs, test_size=0.2, random_state=42)
    val_songs, test_songs = train_test_split(val_test, test_size=0.5, random_state=42)

    # Tokenizer, fit to training set
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_songs)
    total_tokens = len(tokenizer.word_index) + 1
    print('Total tokens: ', total_tokens)  # Not all tokens. Only ones that have some number of minimum appearances.

    # Turn into sequence of integers.
    train_seqs = tokenizer.texts_to_sequences(train_songs)
    val_seqs = tokenizer.texts_to_sequences(val_songs)
    test_seqs = tokenizer.texts_to_sequences(test_songs)

    print(len(train_seqs[0]))

    # Create input sequences and targets.
    def create_song_sequences(vectorized_songs):
        X = []
        y = []
        for song_vect in vectorized_songs:
            generator = TimeseriesGenerator(song_vect, song_vect, 
                                            length=max_len, batch_size=1)
            X.append(np.array([sequence.flatten() for (sequence, _) in generator]))
            y.append(np.array([target.flatten() for (_, target) in generator]))

        return X, y
    
    trn_data = create_song_sequences(train_seqs)
    val_data = create_song_sequences(val_seqs)
    tst_data = create_song_sequences(test_seqs)

    # Pad sequences for consistent input length. Do we have to do this now since they overlap?
    # padded_X = pad_sequences(X, maxlen=max_len, padding='pre')

    return tokenizer, trn_data, val_data, tst_data, total_tokens


def create_model(num_words): # fine tune this architecture
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


# Get lyric data.
lyrics_df = pd.read_csv('country_artist_data/country_lyric_data.csv')
lyrics_df['lyrics'] = lyrics_df['lyrics'].apply(clean_lyrics)

MAX_SEQ = 16 # optimize thissssss
BATCH_SIZE = 32
BUFFER_SIZE = 1000
MODEL_PATH = './streamlined_models/country_model/'

# prep data
tokenizer, train, valid, test, num_words = data_prep(lyrics_df['lyrics'].to_numpy(), MAX_SEQ)

# Save tokenizer
tokenizer_file_path = MODEL_PATH + 'tokenizer.json'
tokenizer_json = tokenizer.to_json()
with open(tokenizer_file_path, 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))

# dataset = Dataset.from_tensor_slices((X, y))
# dataset = dataset.shuffle(buffer_size=1000).batch(BATCH_SIZE)

model = create_model(num_words)

# Train the model
model.fit(, y, epochs=20, verbose=2)

# model.save('./streamlined_models/lyrics_generator_1', save_format='tf')

# seed_text = 'For the love of'
# generated_text = generate_text(model, tokenizer, MAX_SEQ, seed_text, 50)
# print(generated_text)
