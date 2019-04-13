import numpy as np
import pandas as pd
import scholarly
import time

# Add path reference to `src`
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

from src.text_utils import remove_middle_names
from src.utils import flatmap, tqdm_f


def prepare_scholars_df(src: str = 'data/provided_json/data_train.json',
                        dst: str = 'data/scraped/scholars.pkl'
                        ) -> None:
    '''
    Create a dataframe of scholars to be parsed.

    Keyword Arguments:
        src {str} -- Path to training data.
                     (default: {'data/provided_json/data_train.json'})
        dst {str} -- Save path of scholars dataframe.
                     (default: {'data/scraped/scholars.pkl'})

    Raises:
        TypeError -- If args file suffices are incorrect.
        FileExistsError -- If file already exists at `dst`.

    Returns:
        pd.DataFrame -- A dataframe with two columns: 'scholars' and 'profile'.
    '''

    if Path(dst).suffix != '.pkl':
        raise TypeError('`dst` must be a pickle file of extension .pkl')

    if Path(dst).is_dir():
        raise FileExistsError(f'File exists at {dst}')

    # Load and preprocess data
    df = pd.read_json(src)
    df['authors'] = (
        df['authors']
        .apply(lambda authors: [val for d in authors for val in d.values()])
        .apply(lambda names: [remove_middle_names(name) for name in names])
    )

    # Obtain new dataframe consisting of scholars and
    # the number of papers there were involved with
    scholars = (
        flatmap(df['authors'], include_count=True)
        .rename({'authors': 'scholar'}, axis=1)
    )

    # Add empty column for profiles
    scholars['profile'] = np.nan

    # Save file
    scholars.to_pickle(dst)


def scrape(src: str = 'data/scraped/scholars.pkl'):
    '''
    Scrape scholar profile data.

    Keyword Arguments:
        src {str} -- Path to scholars dataframe.
                     (default: {'data/scraped/scholars.pkl'})

    Raises:
        TypeError -- `src` must be a Pickle file.
    '''

    # Get scholar data
    if not Path(src).is_file():
        print('Scholar file is not found. '
              f'A new one will be generated at {src}.')
        prepare_scholars_df()
    if Path(src).suffix != '.pkl':
        raise TypeError('`src` must be a pickle file of extension .pkl')

    df = pd.read_pickle(src)

    # Modify column type to enable insertion of non-float values
    df['profile'] = df['profile'].astype(object)

    rows_to_scrape = df['profile'].isna()

    for i in tqdm_f(is_range=False)(df.index,
                                    desc='scraping profiles',
                                    total=len(df)):
        if not rows_to_scrape[i]:
            continue

        try:
            # Find scholar on Google Scholar
            scholar_name = df.at[i, 'scholar']
            queries = list(scholarly.search_author(scholar_name))

            # If no results, mark profile appropriately
            if not len(queries):
                df.at[i, 'profile'] = 'N/A'

            # Find the exact scholar
            profile_found = False
            for q in queries:
                if remove_middle_names(q.name) == scholar_name:
                    profile_found = True
                    break

            # Get further metadata and mark profile appropriately
            if profile_found:
                df.at[i, 'profile'] = q.fill()
            else:
                df.at[i, 'profile'] = 'N/A'

        except Exception as e:
            print(e)
            df.to_pickle(src)
            time.sleep(60)
        except KeyboardInterrupt:
            print('\nKeyboard interrupt detected! Saving dataframe...')
            df.to_pickle(src)
            sys.exit(0)

    # Save all results at the end of the loop
    print('Scraping complete! Saving dataframe...')
    df.to_pickle(src)


if __name__ == '__main__':
    scrape()
