import pandas as pd
from typing import Sequence, Tuple


def save(scores: Sequence[float],
         dst: str = 'data/coherence_scores.csv'):
    '''
    Save coherence scores into a csv file.

    Arguments:
        scores {Sequence[float]} -- A list of coherence scores.
    '''
    df = pd.DataFrame(scores, columns=['scores'])
    df.to_csv(dst, header=False, index=False)


def load(src: str = 'data/coherence_scores.csv') -> Tuple[float]:
    '''
    Load a list of coherence scores from a CSV file.

    Keyword Arguments:
        src {str} -- CSV file location. (default: {'data/coherence_scores.csv'})

    Returns:
        [Tuple[float]] -- A tuple of coherence scores.
    '''
    return tuple(pd.read_csv(src, names=['scores']).values.tolist())
