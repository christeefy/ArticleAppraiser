import numpy as np
import pandas as pd
import time
from datetime import date
from typing import Sequence, Tuple
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from .utils import tqdm_f, flatmap
tqdm.pandas()


def topic_score(N: int,
                decay: float = 0.05,
                base: float = 10) -> Sequence[float]:
    '''
    Compute the topic score. Refer to Notebook 4 for calculation details.

    Arguments:
        N {int} -- Number of articles in a group.

    Keyword Arguments:
        decay {float}
        -- Decay rate hyperparameter. Between 0 and 1 exclusive. Larger values
           means more weight given to chronological order. (default: {0.05})
        base {float}
        -- Log base hyperparameter. Must be more than zero. Larger value means
           more weight given to popularity of topic. (default: {10})

    Raises:
        ValueError -- If `decay` is not strictly between 0 and 1.
        ValueError -- If base is not strictly positive.

    Returns:
        Sequence[float] -- A list of topic scores for the chronologically
                           sorted articles in the group.
    '''
    if not 0 < decay < 1:
        raise ValueError('Decay has to be between 0 and 1, exclusive.')
    if base <= 0:
        raise ValueError('Base must be strictly positive.')
    return (1 - decay)**np.arange(N) * np.log(N) / np.log(base)


def author_score(authors: pd.Series, scholars: pd.DataFrame) -> pd.Series:
    '''
    Compute the author score. Refer to Notebook 4 for calculation details.

    Arguments:
        authors {pd.Series}
        -- Each element is a list of authors for each document.
        scholars {pd.DataFrame}
        -- A dataframe containing all authors in `authors`.

    Raises:
        TypeError -- `authors` must be a pd.Series
        ValueError -- All authors in `authors` must be a subset of scholars.

    Returns:
        pd.Series -- The author score for corresponding to each document.
    '''
    if not isinstance(authors, pd.Series):
        raise TypeError('Ensure that authors is a pd.Series')

    # Ensure that all authors exist in scholars
    a_set = set(flatmap(authors))
    universe = set(scholars.index)
    if not a_set <= universe:
        raise ValueError('Some authors are not present in `scholars`.')

    # Calculate the average h-index for all authors for each document
    scores = (
        authors
        .progress_apply(lambda names: scholars.loc[names, 'h-index'].values)
        .apply(lambda idxs: [idx for idx in idxs if isinstance(idx, int)])
        .apply(np.mean)
    )

    # Normalize by scores and map it approximately between 0 and 1
    scores /= scores.max()
    scores = scores.apply(lambda x: np.tanh(3 * x))
    return scores


def novelty_score(df: pd.DataFrame,
                  weights: Tuple[float, float] = (0.5, 0.5)) -> pd.Series:
    '''
    Calculate the novelty score.

    Arguments:
        df {pd.DataFrame} -- A dataframe containing the 'topic_score'
                             and the 'author_score'.

    Keyword Arguments:
        weights {tuple} -- The weights to be used for averaging.
                           (default: {(0.5, 0.5)})

    Raises:
        ValueError -- Weights must contain non-negative elements.

    Returns:
        pd.Series -- The topic scores for each document.
    '''
    if any(w < 0 for w in weights):
        raise ValueError('Weights must contain non-negative elements.')

    return (
        df.progress_apply(lambda row: np.average([row['topic_score'],
                                                  row['author_score']],
                                                 weights=weights),
                          axis=1)
    )


def normalize(series: pd.Series,
              min_val: int = 0,
              max_val: int = 1) -> pd.Series:
    '''
    Min-max normalize `series` to range between 0 and 1.

    Arguments:
        series {pd.Series} -- Series to normalize.

    Keyword Arguments:
        min_val {int} -- Minimum value post-normalization. (default: {0})
        max_val {int} -- Maximum value post-normalization. (default: {1})

    Returns:
        pd.Series -- Normalized series.
    '''
    if min_val >= max_val:
        raise ValueError('`min_val` must be strictly smaller than `max_val`.')

    return (
        MinMaxScaler((min_val, max_val))
        .fit_transform(series.values.reshape(-1, 1))
    )


def assign_scores(df: pd.DataFrame,
                  scholars: pd.DataFrame,
                  weights: Tuple[float, float] = (0.5, 0.5),
                  normalized: bool = False
                  ) -> pd.DataFrame:
    '''
    Assign the (1) topic score, (2) author score, and (3) novelty score to
    each article in `df`.

    Arguments:
        df {pd.DataFrame} -- DataFrame containing articles with
                             'topic_score' columns.
        scholars {pd.DataFrame} -- DataFrame containing scholar profiles.

    Keyword Arguments:
        weights {Tuple[float, float]} -- Novelty Score weights.
                                         (default: {(0.5, 0.5)})
        normalized {bool} -- Whether to min-max normalize the Novelty Score.
                             (default: {False})

    Returns:
        pd.DataFrame -- Same as `df` with 'topic_score', 'author_score' and
                        'novelty_score' columns appended.
    '''

    output = []

    # Topic score calculation
    for _, _df in tqdm_f(is_range=False)(df.groupby('top_topics'),
                                         desc='topic score'):
        _df = _df.sort_values('publish_date').reset_index()
        _df['topic_score'] = topic_score(len(_df))
        output.append(_df)
    output = pd.concat(output).set_index('index').sort_values('index')
    output.index.name = None

    # Author score calculation
    print('Calculating author scores')
    time.sleep(0.5)
    output['author_score'] = author_score(output['authors'], scholars)

    # Final score calculation
    print('Calculating final score')
    time.sleep(0.5)
    output['novelty_score'] = novelty_score(output, weights=weights)

    if normalized:
        output['novelty_score'] = normalize(output['novelty_score'])

    return output
