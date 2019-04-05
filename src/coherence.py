import numpy as np
from typing import Union, Optional, Sequence
from tqdm import tqdm, tqdm_notebook

from .utils import in_ipynb, parallelize


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
        self.common_doc_counts = self._get_common_doc_count()
        self._inv_term_sum = 1 / sum(self.doc_freq.values())


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


    def common_doc_count(self,
                         i1: Union[str, int],
                         i2: Union[str, int]) -> int:
        if isinstance(i1, str):
            i1 = self.term_to_id[i1]
        if isinstance(i2, str):
            i2 = self.term_to_id[i2]

        return self.common_doc_counts[i1, i2]


    def _get_common_doc_count(self):
        N_terms = self.doc_term_matrix.shape[1]
        counts = np.empty((N_terms, N_terms), dtype=int)

        counts = (
            np.logical_and(
                np.expand_dims(self.doc_term_matrix, 1),  # Shape (D x 1 x T)
                np.expand_dims(self.doc_term_matrix, -1))  # Shape (D x T x 1)
            .sum(axis=0)
            .squeeze()
        )

        return counts


    def umass(self, i1: Union[str, int], i2: Union[str, int]) -> float:
        return (np.log((self.common_doc_count(i1, i2) + self._inv_term_sum) /
                       self.doc_count(i1)))


    def coherence(self, top_n: Optional[int] = None) -> float:
        if isinstance(top_n, int) and top_n < 2:
            raise AssertionError('`top_n` must be more than 2.')

        # Get terms sorted in descreasing order of frequency
        tqdm_func = tqdm_notebook if in_ipynb() else tqdm
        score = 0.0
        for topic_terms in tqdm_func(self.topic_terms,
                                     desc='calc coherence',
                                     leave=False):
            terms = sorted(topic_terms, key=lambda x: -self.doc_freq[x])

            M = len(terms) if top_n is None else top_n

            for i in range(1, M):
                for j in range(i):
                    score += self.umass(terms[j], terms[i])
        # Get the average score per topic per trial
        score /= (M * (M - 1)) / 2
        score /= len(self.topic_terms)
        return score
