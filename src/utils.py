from math import ceil
import pandas as pd
from typing import Callable, Sequence, Optional, Any, Dict, Iterable
from multiprocessing import Pool, cpu_count
from tqdm import tqdm, tqdm_notebook, trange, tnrange
from functools import partial, update_wrapper


def in_ipynb() -> bool:
    '''
    Check whether running in IPython mode.
    '''

    try:
        get_ipython()
        return True
    except:
        return False


def tqdm_f(is_range: bool = False) -> Callable[[...], ...]:
    '''
    Return a tqdm function depending on the context.
    i.e. Whether it is called in IPython mode and whether it
    is for a range iterable.

    Keyword Arguments:
        is_range {bool} -- Whether it is being used on a
                           range iterable. (default: {False})

    Returns:
        Callable[[...], ...] -- A contextualized tqdm function.
    '''

    if is_range:
        return tnrange if in_ipynb() else trange
    else:
        return tqdm_notebook if in_ipynb() else tqdm


def curry(func: Callable[[Any], Any],
          **kwargs: Dict[Any, Any]) -> Callable[[Any], Any]:
    '''
    Curry the function with **kwargs and update the function __name__.

    Arguments:
        func {Callable[[Any], Any]} -- Function to curry.

    Returns:
        Callable[[Any], Any] -- A curried function with updated
                                function metadata.
    '''

    return update_wrapper(partial(func, **kwargs), func)


def batch(seq: Sequence[Any],
          batch_size: int = 16) -> Iterable[Sequence[Any]]:
    '''
    Convert a `seq` into a sequence of batches.

    Arguments:
        seq {Sequence[Any]} -- A sequence to batch.

    Keyword Arguments:
        batch_size {int} -- Number of items in each batch. (default: {16})

    Returns:
        Iterable[Sequence[Any]] -- A sequence of batches generator.
    '''

    for i in range(ceil(len(seq) / batch_size)):
        yield seq[(batch_size * i):(batch_size * (i + 1))]


def flatmap(series: pd.Series,
            include_count: bool = False) -> pd.DataFrame:
    '''
    Flatmap values in a pd.Series to obtain a list of the unique values.

    Arguments:
        series {pd.Series}
        -- A Series where each row is a list of categorical values.

    Keyword Arguments:
        include_count {bool}
        -- Include a 'freq' column stating the number of times a unique
           value was found in `series`. (default: {False})

    Returns:
        pd.DataFrame -- A DataFrame of unique values.
    '''

    # Flatmap values
    res = (
        series
        .apply(pd.Series)
        .unstack()
        .reset_index(drop=True)
        .dropna()
    )

    # Include the frequency column
    if include_count:
        res = (
            pd.DataFrame(res.value_counts(), columns=['freq'])
            .reset_index()
            .rename({'index': series.name}, axis=1)
        )
    else:
        res = pd.DataFrame(res.unique(), columns=[series.name])

    return res


def parallelize(func: Callable[[...], ...],
                iterable: Sequence,
                n_process: Optional[int] = None,
                chunksize: int = 1,
                leave: bool = True) -> Sequence:
    '''
    Replicates `func` across `n_process` CPUs to process `iterable`.
    Displays progrses using tqdm.

    Arguments:
        func {Callable} -- Function to replicate.
        iterable {Sequence} -- Iterable of arguments to feed into `func`.

    Keyword Arguments:
        n_process {Optional[int]} -- Number of CPUs to replicate func.
                                     If None, use all CPUs. (default: {None})
        chunksize {int} -- Multiprocessing chunksize. Larger values may be
                           beneficial for very long iterables. (default: {1})
        leave {bool} -- Whether to retain completed tqdm progress bar.
                        (default: {True})

    Returns:
        Sequence -- A list of ordered results.
    '''

    if n_process is None:
        n_process = cpu_count()

    with Pool(n_process) as p:
        tqdm_func = tqdm_notebook if in_ipynb() else tqdm

        return list(tqdm_func(p.imap(func, iterable, chunksize=chunksize),
                              desc=func.__name__,
                              total=len(iterable),
                              leave=leave))
