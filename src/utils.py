import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm, tqdm_notebook


def remove_middle_names(name: str) -> str:
    names = name.split()
    return ' '.join([names[0], names[-1]])


def flatmap(series: pd.Series, include_count: bool = False) -> pd.DataFrame:
    res = (
        series
        .apply(pd.Series)
        .unstack()
        .reset_index(drop=True)
        .dropna()
    )

    if include_count:
        res = (
            pd.DataFrame(res.value_counts(), columns=['freq'])
            .reset_index()
            .rename({'index': series.name}, axis=1)
        )
    else:
        res = pd.DataFrame(res.unique(), columns=[series.name])

    return res


def parallelize(func, iterable):
    with Pool(cpu_count()) as p:
        tqdm_func = tqdm_notebook if in_ipynb() else tqdm

        return list(tqdm_func(p.imap(func, iterable),
                              desc=func.__name__,
                              total=len(iterable)))


def in_ipynb() -> bool:
    try:
        get_ipython()
        return True
    except:
        return False
