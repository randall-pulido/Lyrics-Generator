# Lyrics Generator

## Introduction and Motivation

I developed this python project out of my love for listening to, writing, and playing music as well as my interest in further exploring NLP as it has become a very hot topic as of late. Modern technology has contributed to helping artists sharpen their craft and explore the full breadth of their creativity. ML techniques can certainly be utilized for these aims as well, and that was the main goal that I had in mind when I decided to start this project. Not only could I have fun reading computer-generated lyrics, but maybe I could incorporate them into a unique song and write guitar chords to accompany them.

------------------

## Methods

### Data Retrieval

Artist data (such as artist name and genre) is retrieved from [MTV's Top 10,000 Music Artists](https://gist.github.com/mbejda/9912f7a366c62c1f296c#file-10000-mtv-music-artists-page-1-csv), filtered by genre, and fed to the [LyricsGenius](https://lyricsgenius.readthedocs.io/en/master/) which then pulls in lyrics data from a user-specified number of songs for each artist. Data retrieval and further preprocessing, such as cleaning and tokenizing the data (see below), is done in the `helpers.py` file.

### Preprocessing

In order to properly tokenize the song lyrics, certain erroneous data has to be removed from the raw strings delivered by LyricsGenius. For example, advertisements, song structure formatting and labelling, inconsistent whitespace, and irrelevant song production data must be removed from each song's raw lyric data. Removing these from the song lyrics retrieved from the LyricsGenius API is pretty straightforward using Python's regular expression support module ([re](https://docs.python.org/3/library/re.html)), however the songs sometimes have unique inconsistencies which presents another challenge to the data cleaning process. With such a large amount of data being used to train the model, it is difficult to algorithmically pin down where every single inconsistency exists. This merits further investigation into more optimal ways to leverage some other text-filtering module or even using a different or secondary origin from which to retreive lyric data.

Once the raw data is cleaned, the remaining lyrics are tokenized by word.

I took inspiration and adapted code from [this ActiveState article](https://www.activestate.com/blog/how-to-build-a-lyrics-generator-with-python-recurrent-neural-networks/) and [this Machine Learning Mastery article](https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model-in-keras/).

### Model

### Improvements

The project is still a work in progress. Some adjustments that need to be made include:
* Improving the data cleaning process to filter out erroneous data.
    * Described in the [Data Retrieval](#data-retrieval) section above.
* Optimizing the architecture of the LSTM.
* Creating a UI.

------------------

## Results
