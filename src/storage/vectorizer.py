import pickle
from pathlib import Path
from textacy import Vectorizer


def save(vectorizer: Vectorizer, src: str = 'data/nlp/vectorizer.pkl'):
    '''
    Save a Textacy Vectorizer.

    Arguments:
        vectorizer {Vectorizer} -- A vectorizer that was fitted to a corpus.

    Keyword Arguments:
        src {str} -- Save path.
    '''
    if Path(src).suffix != '.pkl':
        raise TypeError('File must be of type .pkl')

    with open(src, 'wb') as f:
        pickle.dump(vectorizer, f)


def load(src: str = 'data/nlp/vectorizer.pkl') -> Vectorizer:
    '''
    Load a fitted Textacy Vectorizer.

    Keyword Arguments:
        src {str} -- Path to vectorizer file.
    '''
    if not Path(src).is_file():
        raise FileNotFoundError('Vectorizer file is not available.')
    if Path(src).suffix != '.pkl':
        raise TypeError('File must be of type .pkl')

    with open(src, 'rb') as f:
        return pickle.load(f)