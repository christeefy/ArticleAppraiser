import numpy as np
import pandas as pd
import textacy
import spacy

import time
from datetime import timedelta

from typing import Optional
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

from src import novelty, storage
from src.text_utils import remove_middle_names
from src.utils import flatmap, parallelize, curry


def parse_dates(x: pd.Series) -> str:
    '''
    Combine 'day', 'month', and 'year' fields into a date string.

    Arguments:
        x {pd.Series} -- Contains 'day', 'month' and 'year' fields.

    Returns:
        str -- A string of format DD/MM/YYYY
    '''

    return f'{x["day"]}/{x["month"]}/{x["year"]}'


def preprocess_text(text: str, nlp) -> str:
    '''
    Text preprocessing that performs:
    1. Stop word removal
    2. Verb removal

    Note:
        Lowercasing is not preemptively here as casing is important to identify
        named entities later down in the pipeline
    '''
    doc = nlp(text)

    # Filter out stop words
    filtered = textacy.extract.words(doc, filter_stops=True, filter_nums=True)

    # Filter out verbs
    filtered = filter(lambda x: x.pos_ != "VERB", filtered)

    return ' '.join(token.text for token in filtered)


def top_topics(model: textacy.TopicModel,
               doc_topic_matrix,
               n_topics: int = 3) -> pd.DataFrame:
    '''
    Return the `n_topics` top topics for each document.

    Arguments:
        model {textacy.TopicModel} -- A trained Topic Model.
        doc_topic_matrix -- A document-topic matrix associated with `model`.

    Keyword Arguments:
        n_topics {int} -- The number of top topics for each document, in
                          descending order. (default: {3})

    Returns:
        pd.DataFrame -- A DataFrame with the 'top_topics' column.
    '''
    return (
        pd.DataFrame(map(lambda x: [tuple(sorted(x[1]))],
                         model.top_doc_topics(doc_topic_matrix, top_n=n_topics)),
                     columns=['top_topics'])
    )


def add_new_scholars_to_df(df: pd.DataFrame,
                           scholars: pd.DataFrame) -> pd.DataFrame:
    '''
    Helper function that adds new scholars found in the validation or
    test `df` into the `scholars` dataframe.

    Arguments:
        df {pd.DataFrame} -- A dataframe for the validation or test set.
        scholars {pd.DataFrame} -- Dataframe containing scholar names
                                   and their profile data.

    Returns:
        pd.DataFrame -- The scholars dataframe extended with new scholars
                        names found in `df`.
    '''
    # Create sets for scholars found in `scholars` and `df` respectively
    universe = set(scholars.index.values.ravel().tolist())
    new_names = set(flatmap(df['authors'])['authors'].values.ravel().tolist())

    # Find new scholar names and create necessary columns with null values
    new_df = pd.DataFrame(new_names - universe, columns=['scholar'])
    for col in ['avg_citedby_2015', 'citedby', 'h-index', 'i10-index']:
        new_df[col] = 'N/A'

    # Append new scholars to `scholars`
    return scholars.append(new_df.set_index('scholar'), sort=False)


def featurize_publish_date(df: pd.DataFrame,
                           months_lookahead: int = 12,
                           prior: Optional[pd.DataFrame] = None
                           ) -> pd.DataFrame:
    '''
    Calculate the modified Topic Score for articles, with papers only able
    to utilize all intra-topic papers with publish dates not exceeding
    `months_lookahead` of said paper.

    Arguments:
        df {pd.DataFrame} -- DataFrame containing 'publish_date' and
                             'top_topics' fields.

    Keyword Arguments:
        months_lookahead {int} -- Intra-topic papers whose publish date does
                                  not exceed this value relative to the paper
                                  are included in the modified Topic Score
                                  calculation. (default: {12})
        prior {Optional[pd.DataFrame]} -- If included, it will be prepended
                                          to `df`. Used for validation and test
                                          sets. (default: {None})

    Returns:
        pd.DataFrame -- DataFrame containing original documents in `df`, with an
                        additional column named `modTopicScore`.
    '''
    if prior is not None:
        # Make a deep copy of `prior` to guarantee that changes are local
        prior = prior.copy()

        # Modify index of `prior` to easily retrieve data of
        # `df`, post-processing
        original_index = df.index
        prior.index = [f'temp{i}' for i in prior.index]
        df = prior.append(df, sort=False)

    # Sort by publish date and add new column
    df = df.sort_values('publish_date')
    df['lookahead'] = np.nan

    # Perform modified Topic Score calculations for each topic group
    for _, _df in df.groupby('top_topics'):
        dates = _df['publish_date']
        for i, (idx, _) in enumerate(_df.iterrows()):
            lookahead_until = (_df.at[idx, 'publish_date'] +
                               timedelta(days=31 * months_lookahead))
            l = len(dates[dates < lookahead_until])
            df.at[idx, 'lookahead'] = novelty.topic_score(l)[i]

    # Recover original entries of `df`
    if prior is not None:
        df = df.loc[original_index]

    # Rename column
    df = df.rename({'lookahead': 'modTopicScore'}, axis=1)

    return df.sort_index()


def load_dataframe(src: str,
                   train_df_dir: str = 'data/dataframes/main_df_scored.pkl',
                   scholars_dir: str = 'data/dataframes/scholars_df.pkl',
                   vectorizer_dir: str = 'data/nlp/vectorizer.pkl',
                   topic_model_dir: str = 'data/nlp/topic_model_21.pkl'
                   ) -> pd.DataFrame:
    '''
    Preprocess raw JSON data for the validation and test datasets in the same
    manner that was done on the training dataset.

    Arguments:
        src {str} -- Path to validation or test JSON data.

    Keyword Arguments:
        train_df_dir {str}
        -- Path to training dataframe. (default: {'data/dataframes/main_df_scored.pkl'})
        scholars_dir {str}
        -- Path to scholars dataframe. (default: {'data/dataframes/scholars_df.pkl'})
        vectorizer_dir {str}
        -- Path to vectorizer model. (default: {'data/nlp/vectorizer.pkl'})
        topic_model_dir {str}
        -- Path to topic model. (default: {'data/nlp/topic_model_21.pkl'})

    Raises:
        TypeError -- `src` must be a JSON file, while all other arguments a Pickle file.
        FileNotFoundError -- If no file exists at provided paths.

    Returns:
        pd.DataFrame -- A processed dataframe ready for modelling.
    '''

    # Check validity of arguments and conver them to Path variables
    for i, var in enumerate((src, train_df_dir, scholars_dir,
                             vectorizer_dir, topic_model_dir)):
        if isinstance(var, str):
            var = Path(var)

        if var.suffix != ('.pkl' if i else '.json'):
            raise TypeError(f"Only {'pickle' if i else 'JSON'} files can be processed.")
        if not scholars_dir.is_file():
            raise FileNotFoundError('Scholar file is not found.')

    # Load relevant data objects
    print('Loading relevent objects...')
    df = pd.read_json(src)
    train_df = pd.read_pickle(train_df_dir)
    scholars = pd.read_pickle(scholars_dir)
    vectorizer = storage.vectorizer.load(vectorizer_dir)
    topic_model = storage.topic_model.load(topic_model_dir)
    nlp = spacy.load('en_core_web_lg')

    # Perform basic preprocessing
    df['authors'] = (
        df['authors']
        .apply(lambda x: [val for d in x for val in d.values()])
        .apply(lambda names: [remove_middle_names(name) for name in names])
    )
    df['publish_date'] = (
        df
        .apply(parse_dates, axis=1)
        .apply(pd.to_datetime)
    )
    df['title+abstract'] = df.apply(lambda row: '.  '.join([row['title'],
                                                            row['abstract']]),
                                    axis=1)

    # Perform topic extraction
    print('Extracting topics...')
    time.sleep(0.5)
    corpus = textacy.Corpus(nlp, texts=(
        df['title+abstract'].
        progress_apply(lambda x: preprocess_text(x, nlp))
        .values
        .tolist()
    ))
    doc_terms = [list(doc.to_terms_list(ngrams=(1, 2, 3),
                                        named_entities=True,
                                        as_strings=True)) for doc in corpus]
    doc_term_matrix = vectorizer.transform(doc_terms)
    doc_topic_matrix = topic_model.transform(doc_term_matrix)
    df['top_topics'] = top_topics(topic_model, doc_topic_matrix, n_topics=3)

    # Perform scoring
    print('Creating scores...')
    time.sleep(0.5)
    scholars = add_new_scholars_to_df(df, scholars)
    df['author_score'] = novelty.author_score(df['authors'], scholars)
    df['score'] = novelty.assign_scores(train_df.append(df, sort=False),
                                        scholars,
                                        normalized=False)[-len(df):]

    return df
