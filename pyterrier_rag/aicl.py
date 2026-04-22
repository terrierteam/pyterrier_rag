"""
AICL - Adaptive In-Context Learning
Implements variable context size for RAG pipelines.

Based on: "One Size Doesn't Fit All: Predicting the Number of Examples 
for In-Context Learning", Chandra et al., ECIR 2025.

Usage:
    from pyterrier_rag.aicl import AICLContextSelector

    aicl = AICLContextSelector(k_max=20)
    aicl.fit(train_retrieved_df, train_labels)

    pipeline = bm25 % 20 >> aicl >> reader
"""

import numpy as np
import pandas as pd

try:
    import pyterrier as pt
    BASE_CLASS = pt.Transformer
except ImportError:
    BASE_CLASS = object

from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class AICLContextSelector(BASE_CLASS):
    """
    Adaptive In-Context Learning (AICL) context size selector.

    For each query, predicts the optimal number of retrieved documents (k)
    to pass to the LLM reader, rather than always using a fixed cutoff.

    A multi-label classifier is trained where each label corresponds to
    whether using k documents leads to a correct answer. At inference,
    the classifier predicts which k values will work, and the highest
    predicted k is selected.

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
            DataFrame with columns: qid, query, score (retrieved docs, grouped by qid).
            Each qid should have up to k_max rows.
        labels : np.ndarray or list of lists
            Shape (n_queries, k_max). labels[i][j] = 1 if using j+1 docs
            gives a correct answer for query i, else 0.

        Returns
        -------
        self
        """
        features = self._extract_features_all(retrieved_df)
        labels_array = np.array(labels)

        # Pad or trim labels to k_max columns
        if labels_array.shape[1] < self.k_max:
            pad = np.zeros((labels_array.shape[0], self.k_max - labels_array.shape[1]))
            labels_array = np.hstack([labels_array, pad])
        else:
            labels_array = labels_array[:, :self.k_max]

        features_scaled = self.scaler.fit_transform(features)
        self.classifier.fit(features_scaled, labels_array)
        self.fitted = True
        return self

    def transform(self, retrieved):
        """
        Filter retrieved documents to the predicted optimal k per query.

        Parameters
        ----------
        retrieved : pd.DataFrame
            Standard PyTerrier retrieved docs DataFrame.

        Returns
        -------
        pd.DataFrame
            Same format, but each query group trimmed to predicted k rows.
        """
        if not self.fitted:
            # Not trained yet — pass through unchanged
            return retrieved

        result_parts = []
        for qid, group in retrieved.groupby('qid', sort=False):
            group_sorted = group.sort_values('score', ascending=False)
            k_pred = self._predict_k_for_query(group_sorted)
            result_parts.append(group_sorted.head(k_pred))

        if not result_parts:
            return retrieved

        return pd.concat(result_parts).reset_index(drop=True)

    def _extract_features_all(self, retrieved_df):
        """Extract feature vectors for all queries in a DataFrame."""
        features = []
        for qid, group in retrieved_df.groupby('qid', sort=False):
            group_sorted = group.sort_values('score', ascending=False)
            features.append(self._query_features(group_sorted))
        return np.array(features)

    def _query_features(self, group):
        """
        Extract numeric features from a group of retrieved docs for one query.

        Features:
        - mean retrieval score
        - std of retrieval scores
        - max score
        - min score
        - score range (max - min)
        - score drop from rank 1 to rank 2 (if available)
        - number of retrieved docs
        - query length in words
        """
        scores = group['score'].values[:self.k_max]

        # Pad scores if fewer than k_max docs
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

        # Also include top-k scores as features
        return base_features + list(scores[:self.k_max])

    def _predict_k_for_query(self, group):
        """Predict optimal k for a single query group."""
        features = self._query_features(group)
        features_scaled = self.scaler.transform([features])
        preds = self.classifier.predict(features_scaled)[0]

        # Find the highest k that is predicted to give correct answer
        valid_ks = [k + 1 for k, p in enumerate(preds) if int(p) == 1]

        if valid_ks:
            return min(max(valid_ks), len(group))  # don't exceed available docs
        else:
            return min(self.fallback_k, len(group))

    def predict_k(self, retrieved_df):
        """
        Public method: return predicted k for each query.

        Parameters
        ----------
        retrieved_df : pd.DataFrame

        Returns
        -------
        dict mapping qid -> predicted k
        """
        if not self.fitted:
            raise RuntimeError("Call .fit() before .predict_k()")

        predictions = {}
        for qid, group in retrieved_df.groupby('qid', sort=False):
            group_sorted = group.sort_values('score', ascending=False)
            predictions[qid] = self._predict_k_for_query(group_sorted)
        return predictions

    @staticmethod
    def build_labels_from_results(retrieved_df, answers_df, answer_col='answers', k_max=20):
        """
        Utility: automatically build training labels by testing each k value.

        For each query, tests k=1..k_max and checks if the top-k docs
        contain words from the gold answer.

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
                # Check if any gold answer appears in top-k docs
                found = any(ans in top_k_text for ans in gold_answers)
                labels_for_query.append(1 if found else 0)

            all_labels.append(labels_for_query)

        return all_labels