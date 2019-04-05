import pytest
import pickle
import numpy as np
from src.coherence import Coherence


@pytest.fixture()
def coherence():
    with open('tests/data/coherence_obj.pkl', 'rb') as f:
        return pickle.load(f)


def test_common_doc_count_method(coherence: Coherence):
    t1, t2 = 'algorithm', 'bayesian'
    v1 = coherence.common_doc_count(t1, t2)
    v2 = coherence.common_doc_count(t2, t1)
    assert v1 == v2


def test_common_doc_count_parser(coherence: Coherence):
    mat = coherence.common_doc_counts
    assert np.all(mat == mat.T)
