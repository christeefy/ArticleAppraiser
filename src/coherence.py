import numpy as np
import pandas as pd
import textacy
from scipy.sparse import csc_matrix
from typing import Union, Optional, List, Dict, Tuple

from multiprocessing import cpu_count

from .utils import tqdm_f, parallelize, curry, flatmap, batch


class Coherence(object):
    '''
    Class to calculate coherence score, based on the UMass function.

    Attributes:
        id_to_term {Dict[int, str]} -- Mapping of term_id to term.
        term_to_id {Dict[str, int]} -- Mapping to term to term_id
        topic_terms {List[Tuple[str, ...]]}
        -- A list containing tuples for each topic. Each tuple contains the
           terms for that topic.
        doc_term_matrix {csc_matrix}
        -- A sparse Boolean matrix of shape (n_docs x n_terms) indicating
           whether a term is present for a given topic.
        freq {Dict[str, int]} -- Number of documents containing each term.
        cofreq {csc_matrix}
        -- A sparse int matrix of shape (n_terms x n_terms) indicating how
           many documents contain both terms i and j, where i and j are the
           column and row indices. Co-frequencies are only calculated for term
           combinations which that will appear in the UMass calculation.
    '''

    def __init__(self,
                 id_to_term: Dict[int, str],
                 doc_term_matrix: Dict[str, int],
                 topic_model: textacy.TopicModel,
                 _chunksize: int = 256) -> None:
        '''Init function.

        Arguments:
            id_to_term {Dict[int, str]}
            -- Mapping from id to term. Obtained from
               trained textacy.Vectorizer.
            doc_term_matrix {Dict[str, int]}
            -- Document term matrix. Transformed from
               trained textacy.Vectorizer.
            topic_model {textacy.TopicModel}
            -- A trained topic model.

        Keyword Arguments:
            _chunksize {int} -- Multiprocessing chunksize. (default: {256})

        Raises:
            AssertionError -- If `topic_model` has not been trained.
        '''
        if not hasattr(topic_model.model, 'components_'):
            raise AssertionError('Topic model has not been trained.')

        self._chunksize = _chunksize  # For multiprocessing

        # Create sorted dictionaries
        self.id_to_term = {i: id_to_term[i]
                           for i in range(max(id_to_term.keys()) + 1)}
        self.term_to_id = {v: k for k, v in self.id_to_term.items()}

        # Add term matrices
        self.topic_terms = self._parse_topic_terms(topic_model)
        self.doc_term_matrix = doc_term_matrix.astype(np.bool).tocsc()

        # Calculate (co)document frequencies
        self.freq = self._parse_doc_freq()
        self.cofreq = self._parse_doc_cofreq()
        self._inv_term_sum = 1 / sum(self.freq.values())


    @staticmethod
    def _calc_cofreq(term_combos: np.ndarray,
                     arr: np.ndarray) -> np.ndarray:
        '''
        Calculate co-term document frequency.
        Optimized to run for batches of numpy arrays.

        Arguments:
            term_combos {List[np.ndarray]}
            -- List of 2-length numpy arrays of ints.
            arr {np.ndarray} -- Boolean array.

        Returns:
            np.ndarray -- An array of ints.
        '''

        return (
            arr[:, term_combos]
            .all(axis=-1)
            .sum(axis=0)
        )


    def doc_freq(self, t: Union[str, int]) -> int:
        '''
        Return the document frequency for a particular term.

        Arguments:
            t {Union[str, int]} -- Term itself or its id.

        Complexity -- Amotized O(1)

        Returns:
            int -- Document frequency of term.
        '''

        if isinstance(t, int):
            t = self.id_to_term[t]

        return self.freq[t]


    def common_doc_freq(self,
                        i1: Union[str, int],
                        i2: Union[str, int]) -> int:
        '''
        Return the co-term document frequency.

        Arguments:
            i1 {Union[str, int]} -- First term or its id.
            i2 {Union[str, int]} -- Second term or its id.

        Complexity -- Amortized O(1)

        Returns:
            int -- Number of documents containing both `i1` and `i2`.
        '''

        if isinstance(i1, str):
            i1 = self.term_to_id[i1]
        if isinstance(i2, str):
            i2 = self.term_to_id[i2]

        if i1 == i2:
            return self.doc_freq(i1)
        else:
            return self.cofreq[i1, i2]


    def coherence(self, top_n: Optional[int] = None) -> float:
        '''
        Calculate the coherence score using the UMass metric.
        Memoization-enabled (i.e. computation skip if this
        function is called again.)

        Keyword Arguments:
            top_n {Optional[int]} -- Top N terms in each topic to include
                                     in calculation. (default: {None})

        Raises:
            AssertionError -- `top_n` has to be more than 2.

        Returns:
            float -- A coherence score for the given number of topics.
        '''
        # Skip computations if coherence score was calculated before
        if hasattr(self, '_coh'):
            return self._coh

        if isinstance(top_n, int) and top_n < 2:
            raise AssertionError('`top_n` must be more than 2.')

        # Get a pd.Series of the counts of each term combo
        combo_counts = self._unique_term_combinations(top_n=top_n,
                                                      counts=True)
        combos, counts = map(np.array, zip(*combo_counts.iteritems()))

        # Calculate the numerator and denominator for the
        # UMass function as numpy arrays
        num = self.cofreq.toarray()[tuple(combos.T)]
        denom = np.array(list(map(
            lambda x: self.doc_freq(self.id_to_term[x[0]]), combos)
        ))

        # Calculate the coherence score based on the UMass metrics
        # The average is taken relative to the number of term combinations and
        # the numer of topics
        score = (counts * np.log((num + self._inv_term_sum) / denom)).sum()
        score /= counts.sum()

        # Save results for memoization
        self._coh = score
        return score


    def _parse_topic_terms(self,
                           topic_model: textacy.TopicModel
                           ) -> List[Tuple[str, ...]]:
        '''
        Create the topic terms from the topic model.

        Arguments:
            topic_model {textacy.TopicModel} -- A trained textacy Topic Model.

        Returns:
            List[Tuple[str, ...]] -- A list of terms for each topic.
        '''
        topic_components = topic_model.model.components_
        topic_terms = []
        for comp in topic_components:
            num_nonzeros = len(comp.nonzero()[0])
            nonzero_terms = np.argsort(comp)[::-1][:num_nonzeros]
            topic_term = tuple(self.id_to_term[i] for i in nonzero_terms)
            topic_terms.append(topic_term)
        return topic_terms


    def _parse_doc_freq(self) -> Dict[str, int]:
        '''
        Create a dictionary containing the document
        frequency for each term.
        '''

        freq = (
            np.array(
                self.doc_term_matrix
                .sum(axis=0))
            .squeeze()
        )

        return dict(zip(self.id_to_term.values(), freq))


    def _parse_doc_cofreq(self) -> csc_matrix:
        '''
        Create a sparse matrix of co-term document frequency.
        Only possible term combinations values are filled, including
        transposed values.

        Note:
            Taking advantage of the sparsity of doc_term_matrix DID NOT HELP.
            Overall, NumPy operations are way more optimized. The original
            issue of exorbitant RAM usage was mitigated by limiting the search
            space and batching, as listed below.

            This function has been optimized as follows:
                (1) Limiting the term combinations to those that appear in
                    `self.topic_terms`.
                (2) Sorting term combinations and removing duplicates.
                (3) Batch term combinations + multiprocessing
                (4) Parallelize computations using numpy

            Overall, this eliminated the enormous RAM requirement and made the
            object instatiation + coherence score calculation to be less than
            a minute on the full dataset.

        Returns:
            csc_matrix -- A co-term document frequency of shape
                          (n_terms x n_terms)
        '''

        combos = list(self._unique_term_combinations())

        # Create collector variables
        data = [None] * len(combos)
        rows = [None] * len(combos)
        cols = [None] * len(combos)

        # Find co-occurence frequencies
        cofreqs = parallelize(curry(Coherence._calc_cofreq,
                                    arr=self.doc_term_matrix.toarray()),
                              list(batch(combos, self._chunksize)),
                              leave=False,
                              chunksize=self._chunksize)
        # Flatmap into a single array
        cofreqs = np.concatenate(cofreqs)

        # Store information for cofrequencies that are non-zero
        n = 0
        for (id1, id2), cofreq in zip(combos, cofreqs):
            if cofreq:
                data[n] = cofreq
                rows[n], cols[n] = id1, id2
                n += 1

        # Create sparse cofreq matrix
        n_terms = self.doc_term_matrix.shape[1]
        cofreq_mat = csc_matrix((data[:n], (rows[:n], cols[:n])),
                                shape=(n_terms, n_terms))

        # Fill in values at transposed positions
        cofreq_mat += cofreq_mat.T

        return cofreq_mat


    def _unique_term_combinations(self,
                                  top_n: Optional[int] = None,
                                  sort: bool = True,
                                  counts: bool = False
                                  ) -> Union[pd.Series, set]:
        '''
        Obtain a unique set of all possible 2-term combinations arising
        from self.topic_terms.

        Keyword Arguments:
            top_n {Optional[int]}
            -- Limit terms to the top N most common terms in each topic.
               If None, use all terms. (default: {None})
            sort {bool}
            -- Whether to sort the term combination. (default: {True})
            counts {bool}
            -- Whether to count the number of occurence for each unique
               term combination (default: {False})

        Returns:
            Union[pd.Series, set]
            -- Returns a set if count is False, otherwise it returns a
               pd.Series with the freq as values and unique term combinations
               as the index.
        '''

        # Get number of unique (w_i, w_j) topic combination IDs
        combos = []
        for topic_terms in self.topic_terms:
            M = (len(topic_terms)
                 if top_n is None else
                 min(top_n, len(topic_terms)))

            for i in range(1, M):
                # Get the term id corresponding to the i-th term of
                # `self.topic_matrix`
                id_i = self.term_to_id[topic_terms[i]]

                for j in range(i):
                    # Do the same for the j-th term
                    id_j = self.term_to_id[topic_terms[j]]

                    # Append the sorted combination
                    # Sorting is done to reduce downstream calculations since
                    # cofreq(w_i, w_j) == cofreq(w_j, w_i)
                    combination = (tuple(sorted((id_i, id_j)))
                                   if sort
                                   else (id_i, id_j))
                    combos.append(combination)

        if counts:
            return pd.Series(combos).value_counts()
        else:
            return set(combos)
