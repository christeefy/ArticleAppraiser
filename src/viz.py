import numpy as np
import pandas as pd
import plotly.offline as pyo
import plotly.graph_objs as go
from textacy import TopicModel

from typing import Sequence, List, Dict, Union

from .utils import in_ipynb

# Enable notebook mode if functions are called from Jupyter
if in_ipynb():
    pyo.init_notebook_mode()


def coherence_scores(scores: Sequence[float]) -> None:
    '''
    Create an interactive scatterplot to visualize the coherence scores to
    select the best number of topic clusters.

    Arguments:
        scores {Sequence[float]} -- A list of coherence scores.
    '''

    trace = go.Scatter(x=list(range(1, len(scores) + 1)), y=scores)
    data = [trace]
    pyo.iplot(data)


def top_terms(model: TopicModel,
              id_to_term: Dict[int, str],
              topics: Union[Sequence[int], int]=-1,
              top_n: int = 10) -> pd.DataFrame:
    '''
    Return the `top_n` terms for each topic as a pandas DataFrame.

    Arguments:
        model {TopicModel} -- A trained textacy.TopicModel
        id_to_term {Dict[int, str]} -- A mapping from ids to terms.
        Can be obtained from a textacy.Vectorizer

    Keyword Arguments:
        topics {Union[Sequence[int], int]}
            -- Topics to return. If -1 (default), return all topics.
        top_n {int} -- Number of top terms to return. (default: {10})

    Returns:
        pd.DataFrame -- A dataframe with columns
                        ['Topic', 'Term 1', ..., 'Term N'].
    '''
    # Flat map topics
    # e.g. (topic0, (term1, ..., term10)) -> (topic0, term1, ..., term10)
    top_terms_by_topic = (
        map(lambda x: (x[0], *x[1]),
            model.top_topic_terms(id_to_term, topics=topics, top_n=top_n))
    )

    df = (pd.DataFrame(top_terms_by_topic,
                       columns=(
                           ['Topic'] +
                           [f'Term {n + 1}' for n in range(top_n)]))
          .set_index('Topic'))

    return df


def top_topics_in_doc(doc_id: int,
                      df: pd.DataFrame,
                      topics_df: pd.DataFrame) -> pd.DataFrame:
    '''
    Retrieve the top topics for a document.

    Arguments:
        doc_id {int} -- Id of documents to retrive.
        df {pd.DataFrame}
        -- Main dataframe containing `top_topics` as column.
        topics_df {pd.DataFrame}
        -- A dataframe containing the outputs of the `viz.top_terms` function.

    Returns:
        pd.DataFrame -- Same dataframe schema as `topics_df`.
    '''
    if 'top_topics' not in df.columns:
        raise KeyError("`df` must contain 'top topics'.")
    if not isinstance(doc_id, int):
        raise TypeError("`doc_id must be an integer.")

    indices = list(df.at[doc_id, 'top_topics'])
    return topics_df.loc[indices]


def topic_trends(df: pd.DataFrame, top_n: int = 10) -> None:
    '''
    Plot the trends for each unique topic combination.

    Arguments:
        df {pd.DataFrame}
        -- A dataframe of schema (unique_topic_combinations x year)
           containing counts of journal articles.

    Keyword Arguments:
        top_n {int} -- Number of top topic combinations to plot. (default: {10})
    '''

    # Filter for the `top_n` topics in df
    docs_per_topic = np.argsort(df.sum(axis=1))[:-top_n - 1:-1]
    df = df.iloc[docs_per_topic]

    data = []
    for topic, row in df.iterrows():
        trace = go.Scatter(x=row.index, y=row.cumsum(), name=str(topic))
        data.append(trace)

    # Create layout and plot figure
    layout = go.Layout(showlegend=True)
    fig = go.Figure(data, layout)
    pyo.iplot(fig)
