import numpy as np
from typing import Union, Tuple
from tqdm import trange, tnrange, tqdm, tqdm_notebook

from .utils import in_ipynb


class Coherence(object):
    def __init__(self, id_to_term, doc_term_matrix, topic_model):
        if not hasattr(topic_model.model, 'components_'):
            raise AssertionError('Topic model has not been trained.')

        # Create sorted dictionaries
        self.id_to_term = {i: id_to_term[i]
                           for i in range(max(id_to_term.keys()) + 1)}
        self.term_to_id = {v: k for k, v in self.id_to_term.items()}

        self.topic_terms = self._parse_topic_terms(topic_model)
        self.doc_term_matrix = doc_term_matrix.toarray().astype(np.bool)
        self.doc_freq = self._parse_doc_freq()


    def _parse_topic_terms(self, topic_model):
        topic_components = topic_model.model.components_
        topic_terms = []
        for comp in topic_components:
            num_nonzeros = len(comp.nonzero()[0])
            nonzero_terms = np.argsort(comp)[::-1][:num_nonzeros]
            topic_term = tuple(self.id_to_term[i] for i in nonzero_terms)
            topic_terms.append(topic_term)
        return topic_terms


    def _parse_doc_freq(self) -> dict:
        doc_freq = self.doc_term_matrix.sum(axis=0)
        return dict(zip(self.id_to_term.values(), doc_freq))


    def doc_count(self, t: Union[str, int]) -> int:
        if isinstance(t, int):
            t = self.id_to_term[t]

        return self.doc_freq[t]


    def common_doc_count(self, i1: Union[str, int], i2: Union[str, int]) -> int:
        if isinstance(i1, str):
            i1 = self.term_to_id[i1]
        if isinstance(i2, str):
            i2 = self.term_to_id[i2]

        return (np.logical_and(self.doc_term_matrix[:, i1],
                               self.doc_term_matrix[:, i2])
                .sum())


    def umass(self, i1: Union[str, int], i2: Union[str, int]) -> float:
        return np.log((self.common_doc_count(i1, i2) + 1) / self.doc_count(i1))


    def coherence(self) -> float:
        # Get terms sorted in descreasing order of frequency
        tqdm_func = tqdm_notebook if in_ipynb() else tqdm
        score = 0.0
        for topic_terms in tqdm_func(self.topic_terms, leave=False):
            terms = sorted(topic_terms, key=lambda x: -self.doc_freq[x])
            print(terms)

            for i in trange(len(terms), leave=False):
                for j in range(i, len(terms)):
                    score += self.umass(terms[i], terms[j])
        return score
