"""
AICL - Adaptive In-Context Learning
Implements variable context size for RAG pipelines.

Based on: Chandra, M., Ganguly, D., Ounis, I. (2025).
One Size Doesn't Fit All: Predicting the Number of Examples
for In-Context Learning. ECIR 2025.

.. cite.dblp:: conf/ecir/ChandraGO25

Usage:
    from pyterrier_rag.aicl import AICLContextSelector

    aicl = AICLContextSelector(k_max=20)
    aicl.fit(train_retrieved_df, train_labels)

    pipeline = bm25 % 20 >> aicl >> reader
"""

import numpy as np
import pandas as pd
import pyterrier as pt
import pyterrier_alpha as pta

from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class AICLContextSelector(pt.Transformer):
    """
    Adaptive In-Context Learning (AICL) context size selector.

    For each query, predicts the optimal number of retrieved documents (k)
    to pass to the LLM reader, rather than always using a fixed cutoff.

    A multi-label classifier is trained where each label corresponds to
    whether using k documents leads to a correct answer. At inference,
    the classifier predicts which k values will work, and the highest
    predicted k is selected.

    .. cite.dblp:: conf/ecir/ChandraGO25

    Parameters
    ----------
    k_max : int
        Maximum number of documents to consider. Default: 20.
    classifier : sklearn estimator, optional
        Multi-label classifier. Defaults to MultiOutputClassifier(LogisticRegression()).
    fallback_k : int, optional
        k to use if classifier predicts no valid k. Defaults to k_max.
    """

    def __init__(self, k_max=20, classifier=None, fallback_k=None):
        self.k_max = k_max
        self.fallback_k = fallback_k if fallback_k is not None else k_max
        self.classifier = classifier or MultiOutputClassifier(
            LogisticRegression(max_iter=1000, random_state=42)
        )
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, retrieved_df, labels):
        """
        Train the AICL classifier.

        Parameters
        ----------
        retrieved_df : pd.DataFrame
            DataFrame with columns: qid, query, score, docno.
            Each qid should have up to k_max rows, sorted by score descending.
        labels : np.ndarray or list of lists
            Shape (n_queries, k_max). labels[i][j] = 1 if using j+1 docs
            gives a correct answer for query i, else 0.

        Returns
        -------
        self

        Example
        -------
        >>> aicl = AICLContextSelector(k_max=3)
        >>> aicl.fit(train_retrieved_df, train_labels)
        >>> pipeline = bm25 % 20 >> aicl >> reader
        """
        features = self._extract_features_all(retrieved_df)
        labels_array = np.array(labels)

        if labels_array.shape[1] < self.k_max:
            pad = np.zeros((labels_array.shape[0], self.k_max - labels_array.shape[1]))
            labels_array = np.hstack([labels_array, pad])
        else:
            labels_array = labels_array[:, :self.k_max]

        features_scaled = self.scaler.fit_transform(features)
        self.classifier.fit(features_scaled, labels_array)
        self.fitted = True
        return self

    @pta.transform.by_query()
    def transform(self, retrieved):
        """
        Filter retrieved documents to the predicted optimal k per query.

        Parameters
        ----------
        retrieved : pd.DataFrame
            Standard PyTerrier retrieved docs DataFrame.
            Must contain columns: qid, docno, score.

        Returns
        -------
        pd.DataFrame
            Same format, but trimmed to predicted k rows for each query.
        """
        if len(retrieved) == 0:
            return retrieved

        if not self.fitted:
            raise RuntimeError(
                "AICLContextSelector must be fitted before calling transform(). "
                "Call .fit(retrieved_df, labels) first."
            )

        group_sorted = retrieved.sort_values('score', ascending=False)
        k_pred = self._predict_k_for_query(group_sorted)

        if 'rank' in group_sorted.columns:
            return group_sorted[group_sorted['rank'] < k_pred]
        return group_sorted.head(k_pred)

    def _extract_features_all(self, retrieved_df):
        features = []
        for qid, group in retrieved_df.groupby('qid', sort=False):
            group_sorted = group.sort_values('score', ascending=False)
            features.append(self._query_features(group_sorted))
        return np.array(features)

    def _query_features(self, group):
        scores = group['score'].values[:self.k_max]

        if len(scores) < self.k_max:
            scores = np.pad(scores, (0, self.k_max - len(scores)), constant_values=0)

        query_len = 0
        if 'query' in group.columns:
            query_len = len(str(group['query'].iloc[0]).split())

        score_drop = scores[0] - scores[1] if len(scores) > 1 else 0.0

        base_features = [
            float(np.mean(scores)),
            float(np.std(scores)),
            float(np.max(scores)),
            float(np.min(scores)),
            float(np.max(scores) - np.min(scores)),
            float(score_drop),
            float(min(len(group), self.k_max)),
            float(query_len),
        ]

        return base_features + list(scores[:self.k_max])

    def _predict_k_for_query(self, group):
        features = self._query_features(group)
        features_scaled = self.scaler.transform([features])
        preds = self.classifier.predict(features_scaled)[0]

        valid_ks = [k + 1 for k, p in enumerate(preds) if int(p) == 1]

        if valid_ks:
            return min(max(valid_ks), len(group))
        else:
            return min(self.fallback_k, len(group))

    def predict_k(self, retrieved_df):
        """
        Return predicted k for each query.

        Parameters
        ----------
        retrieved_df : pd.DataFrame

        Returns
        -------
        dict mapping qid -> predicted k
        """
        if not self.fitted:
            raise RuntimeError(
                "AICLContextSelector must be fitted before calling predict_k(). "
                "Call .fit(retrieved_df, labels) first."
            )

        predictions = {}
        for qid, group in retrieved_df.groupby('qid', sort=False):
            group_sorted = group.sort_values('score', ascending=False)
            predictions[qid] = self._predict_k_for_query(group_sorted)
        return predictions

    @staticmethod
    def build_labels_from_results(retrieved_df, answers_df, answer_col='answers', k_max=20):
        """
        Utility: build training labels by testing textual inclusion for each k.

        For each query and each value of k (1..k_max), checks whether any
        gold answer string appears as a substring in the concatenated text
        of the top-k retrieved documents (textual inclusion check).
        Labels are 1 if the answer is found in top-k docs, else 0.

        Parameters
        ----------
        retrieved_df : pd.DataFrame
            Retrieved docs with qid, text columns.
        answers_df : pd.DataFrame
            Gold answers with qid and answer column.
        answer_col : str
            Column name for answers in answers_df.
        k_max : int

        Returns
        -------
        list of lists: labels[i] is a list of 0/1 for k=1..k_max
        """
        answer_map = {}
        for _, row in answers_df.iterrows():
            qid = row['qid']
            ans = row[answer_col]
            if isinstance(ans, list):
                answer_map[qid] = [a.lower() for a in ans]
            else:
                answer_map[qid] = [str(ans).lower()]

        all_labels = []
        for qid, group in retrieved_df.groupby('qid', sort=False):
            group_sorted = group.sort_values('score', ascending=False)
            labels_for_query = []
            gold_answers = answer_map.get(qid, [])

            for k in range(1, k_max + 1):
                top_k_text = ' '.join(
                    group_sorted.head(k)['text'].fillna('').str.lower().tolist()
                )
                found = any(ans in top_k_text for ans in gold_answers)
                labels_for_query.append(1 if found else 0)

            all_labels.append(labels_for_query)

        return all_labels