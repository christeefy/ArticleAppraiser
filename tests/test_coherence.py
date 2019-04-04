import pytest
import pickle
from src.coherence import Coherence


@pytest.fixture()
def coherence():
    with open('tests/data/coherence_obj.pkl', 'rb') as f:
        return pickle.load(f)


def test_both_present_method(coherence: Coherence):
    t1, t2 = 'algorithm', 'bayesian'
    v1 = coherence.common_doc_count(t1, t2)
    v2 = coherence.common_doc_count(t2, t1)
    assert v1 == v2
