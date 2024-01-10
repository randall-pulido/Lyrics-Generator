import pandas as pd
import numpy as np
import re
import lyricsgenius
import string

GENIUS_ACCESS_TOKEN = 'Kt4TThKkWKXuV43ItLTV_po3fUB9WDIsz5pvWc-HMI-nATTGgLvL1i7ZXF1KD705'

def get_artists_by_genre(filepath, genres=['country'], start_row=0,
                         artist_col_name='name', genre_col_name='genre', num_artists=20):
    '''Searches for artists based on genre.

    This function finds artists within a .csv file that make a particular kind 
    of music based on their labeled genre. Multiple genres can be passed into 
    the function in the form of a list if more than one genre is of interest. 
    Note: The file is scanned in sequential order, so the distribution of 
    artists among the desired genres will likely be skewed.
    The file contents should be organized such that the artist and genre data 
    may be easily accessible, although column labels may be specified by the
    `artist_col_name` and `genre_col_name` arguments (by deafult these are set
    to `name` and `genre` respectively). A pandas dataframe containing the 
    artists (under `artist` column) and their genre (under `genre` column) is 
    returned. 

    Args:
        filepath: Path to the .csv file.
        genres: List of genres to search for.
        artist_col_name: Name of the column within the .csv file that contains
            artist names.
        genre_col_name: Name of the column within the .csv file that contains 
            genres.
        num_artists: Number of artists to search for.

    Returns:
        A pandas dataframe containing artists and genres.
    
    '''
    df = pd.read_csv(filepath, usecols=[artist_col_name, genre_col_name], skiprows=range(1,start_row), nrows=num_artists) # , skiprows=range(1,100)
    df.rename(columns={artist_col_name:'artist', genre_col_name:'genre'}, inplace=True)
    df['genre'] = df['genre'].str.lower()
    df = df.loc[df['genre'].isin(genres)]
    print(df.size)
    return df


def get_lyrics(artists, max_songs=5):
    '''Searches for artist song lyrics from the lyricsgenius client.

    This function takes in a list of strings of artist names and finds (up to 
    `max_songs`) songs performed by that artist using the lyricsgenius API:
    https://github.com/johnwmillr/LyricsGenius.

    Args:
        artists: A list of artist names.
        max_songs: The maximum number of songs to search for for each artist.
    
    Returns:
        A pandas dataframe containing artists, song titles, and song lyrics.

    '''
    genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN)
    artist_array, song_array, lyrics_array = np.empty(0), np.empty(0), np.empty(0)
    for artist in artists:
        try:
            genius_artist = genius.search_artist(artist, max_songs=max_songs, sort='popularity', 
                                                 get_full_info=False, include_features=True)
        except:
            continue
        else:
            songs = genius_artist.songs
            artist_array = np.append(artist_array, [artist] * len(songs))
            for song in songs:
                song_array = np.append(song_array, song.title)
                lyrics_array = np.append(lyrics_array, song.to_text()) 
                # ^^^ removed clean_lyrics function from around song.to_text() 
                # so that we can save raw lyrics. Use LLM to clean? 
    return pd.DataFrame(np.stack((artist_array, song_array, lyrics_array)).T, 
                       columns=['artist', 'song', 'lyrics'])

def clean_lyrics(lyrics_text):
    '''Cleans song lyrics so that they may be more easily tokenized.
    
    Given a string representation of song lyrics, this function cleans the 
    lyrics by removing the following substrings:
        - All text in the first line of lyrics to remove irrelevant information
            such as song title, number of contributors, etc.
        - Text inside of brackets and parentheses (Ex: `[Intro]`).
        - Unnecessary new line characters.
        - Trailing `Embed` text at the very end of the lyrics.

    Args:
        lyrics_text: String representation of a song's lyrics from lyricsgenius.
    
    Returns:
        A string of the cleaned song lyrics.

    TODO: Improve the removal of advertising text within lyrics.

    '''
    try:
        lyrics_text = lyrics_text.split('\n\n',1)[1]
    except IndexError:
        lyrics_text = lyrics_text.split('\n',1)[1]
    lyrics_text = re.sub('you might also like', ' ', lyrics_text)
    lyrics_text = re.sub('see.*?live', '', lyrics_text)
    lyrics_text = re.sub('get tickets as low as \$?\d+\b?', ' ', lyrics_text)
    lyrics_text = re.sub('[\(\[].*?[\)\]]', '', lyrics_text)
    lyrics_text = re.sub('\n\n\n|\n\n', '\n', lyrics_text)
    lyrics_text = re.sub('\n', ' \n ', lyrics_text)
    lyrics_text = lyrics_text.replace('  ', ' ')
    if lyrics_text.endswith('Embed'):
        lyrics_text = lyrics_text[:-5]
    lyrics_text = re.sub('\.|\,|\!|\?|\-|', '', lyrics_text)
    return lyrics_text.lower()

def split_lyric_data(lyric_data):
    '''Splits text data into tokens by spaces. Preserves newline characters.

    Args:
        lyric_data: String we wish to split by spaces.
    
    Returns:
        A list of tokens retreived from the input text.
    
    '''
    split_data = []
    for song_lyrics in lyric_data:
        split_data += [token for token in song_lyrics.split(' ') 
                       if token.strip != '' or token == '\n']
    return split_data

def get_frequencies(lst):
    '''Get the frequencies of items within a list.

    Args:
        lst: A list of items.
    
    Returns:
        A dictionary with (list item, item frequency) as (key, value).
    
    '''
    frequencies = {}
    for token in lst:
        frequencies[token] = frequencies.get(token, 0) + 1
    return frequencies

def sequence_tokens(split_data, uncommon_tokens, seq_length=5):
    '''Partitions a list into sublists that do not contains any of `uncommon_tokens` 
    with length `seq_length`. The first `seq_length` - 1 elements of the sublists
    are saved to the first resultant list of the returned tuple and the last element of 
    the sublists are saved to the second resultant list of the returned tuple.

    Args:
        split_data: List to be partitioned.
        uncommon_tokens: Set with which to filter `split_data`.
        seq_length: The length of the sequences to partition.

    Returns:
        A 2-tuple. The first element of the tuple is a list of sequences that 
        were partitioned from the given list, excluding the last element of each 
        sequence. The second element of the tuple is a list of the last elements 
        of each sequence.
    
    '''
    valid_seqs = []
    end_seq_words = []
    for i in range(len(split_data) - seq_length): 
        end_slice = i + seq_length + 1
        if len(set(split_data[i:end_slice]).intersection(uncommon_tokens)) == 0:
            valid_seqs.append(split_data[i: i + seq_length])
            end_seq_words.append(split_data[i + seq_length])
    return valid_seqs, end_seq_words
