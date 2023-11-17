from sklearn.model_selection import train_test_split
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from helpers import *
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM, Bidirectional, Embedding

# Process data into dataframe.
path1 = '/Users/randallpulido/Desktop/ML/lyrics_generator/artist_data/top_10000_artists/10000-MTV-Music-Artists-page-1.csv'
artist_genre_df = get_artists_by_genre(path1, num_artists=2)
lyrics_df = get_lyrics(artist_genre_df['artist'].tolist(), max_songs=1)
merged = lyrics_df.merge(artist_genre_df, on='artist', how='left')

# Tokenize lyric data.
tokenized_lyrics = split_lyric_data(merged['lyrics'])
print('Total tokens: ', len(tokenized_lyrics))

# Get token frequencies.
frequencies = get_frequencies(tokenized_lyrics)

# Get set of words and uncommon words.
MIN_FREQUENCY = 7
uncommon_words = set([key for key in frequencies.keys() if frequencies[key] < MIN_FREQUENCY])
words = sorted(set([key for key in frequencies.keys() if frequencies[key] >= MIN_FREQUENCY]))
num_words = len(words)
word_indices = dict((w, i) for i, w in enumerate(words))
indices_word = dict((i, w) for i, w in enumerate(words))
print('Words with less than {} appearances: {}'.format( MIN_FREQUENCY, len(uncommon_words)))
print('Words with more than {} appearances: {}'.format( MIN_FREQUENCY, len(words)))

# Partition data into sequences of tokens and end tokens.
MIN_SEQ = 5
sequences = sequence_tokens(tokenized_lyrics, uncommon_words, seq_length=MIN_SEQ)
valid_seqs, end_tokens = sequences
print('Valid sequences of size {}: {}'.format(MIN_SEQ, len(valid_seqs)))

# Split data into train and test.
X_train, X_test, y_train, y_test = train_test_split(sequences[0], end_tokens, test_size=0.02, random_state=42)

def generator(sentence_list, next_word_list, batch_size):
    '''Data generator.
    
    Yields `batch_size` number of sequences from `sentence_list` and `next_word_list`
    stored as the word's corresponding index from `word_indices` set.

    '''
    index = 0
    while True:
        x = np.zeros((batch_size, MIN_SEQ), dtype=np.int32)
        y = np.zeros((batch_size), dtype=np.int32)
        for i in range(batch_size):
            for t, w in enumerate(sentence_list[index % len(sentence_list)]):
                x[i, t] = word_indices[w]
            y[i] = word_indices[next_word_list[index % len(sentence_list)]]
            index = index + 1
        yield x, y


# Functions from keras-team/keras/blob/master/examples/lstm_text_generation.py
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    examples_file.write('\n----- Generating text after Epoch: %d\n' % epoch)
    # Randomly pick a seed sequence
    seed_index = np.random.randint(len(X_train+X_test))
    seed = (X_train+X_test)[seed_index] # get random sample from train + test
   
    for diversity in [0.3, 0.4, 0.5, 0.6, 0.7]:
        sentence = seed # initial sequence taken from train/test data
        examples_file.write('----- Diversity:' + str(diversity) + '\n')
        examples_file.write('----- Generating with seed:\n"' + ' '.join(sentence) + '"\n')
        examples_file.write(' '.join(sentence))
        for i in range(50): # generating 50 words atm TODO: make this an input.
            x_pred = np.zeros((1, MIN_SEQ)) # initialize empty numpy vector for predicted sequence. 
            # since newline character included in possible words, MIN_SEQ doesn't matter too much
            # in terms of final structure of generated song lyrics.
            for t, word in enumerate(sentence):
                x_pred[0, t] = word_indices[word] # set pred values equal to 
                # initial sequence indices (from dictionary)
            preds = model.predict(x_pred, verbose=0)[0] # get predictions for 
            # given sequence in form of indices
            next_index = sample(preds, diversity) # sample a single index 
            # TODO: worth it to have a variable/random diversity? or potentially optimize?
            next_word = indices_word[next_index] # get next word via index
 
            sentence = sentence[1:] # shift sentence over by one word
            sentence.append(next_word) # append new word to sentence
 
            examples_file.write(" "+next_word)
        examples_file.write('\n')
    examples_file.write('='*80 + '\n')
    examples_file.flush()

def create_model():
    print('Build model...')
    model = Sequential()
    model.add(Embedding(input_dim=len(words), output_dim=1024))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dense(len(words)))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model

BATCH_SIZE = 32
model = create_model()
model.summary()
file_path = "./checkpoints/LSTM_LYRICS-epoch{epoch:03d}-words%d-sequence%d-minfreq%d-" \
           "loss{loss:.4f}-acc{accuracy:.4f}-val_loss{val_loss:.4f}-val_acc{val_accuracy:.4f}" % \
           (len(words), MIN_SEQ, MIN_FREQUENCY)
checkpoint = ModelCheckpoint(file_path, monitor='val_accuracy', save_best_only=True)
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=20)
callbacks_list = [checkpoint, print_callback, early_stopping]
examples_file = open('examples.txt', "w")
model.fit(generator(X_train, y_train, BATCH_SIZE),
                   steps_per_epoch=int(len(valid_seqs)/BATCH_SIZE) + 1,
                   epochs=20,
                   callbacks=callbacks_list,
                   validation_data=generator(X_test, y_train, BATCH_SIZE),
                   validation_steps=int(len(y_train)/BATCH_SIZE) + 1)

def predict():

    pass