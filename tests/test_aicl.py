"""
Tests for AICL adaptive context selector.
"""
import pandas as pd
import numpy as np
import pytest
from pyterrier_rag.aicl import AICLContextSelector


def make_dummy_data():
    df = pd.DataFrame({
        'qid':   ['q1','q1','q1','q2','q2','q2','q3','q3','q3'],
        'query': ['what is python']*3 + ['who is einstein']*3 + ['what is java']*3,
        'score': [0.9, 0.7, 0.5,  0.8, 0.6, 0.3,  0.7, 0.5, 0.2],
        'text':  ['python is language','used for coding','easy to learn',
                  'einstein physicist','born in germany','e=mc2',
                  'java is language','runs on jvm','object oriented']
    })
    labels = [[0,1,1], [1,0,0], [0,0,1]]
    return df, labels


def test_import():
    from pyterrier_rag.aicl import AICLContextSelector
    assert AICLContextSelector is not None


def test_fit_transform():
    df, labels = make_dummy_data()
    aicl = AICLContextSelector(k_max=3)
    aicl.fit(df, labels)
    result = aicl.transform(df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_output_has_fewer_or_equal_rows():
    df, labels = make_dummy_data()
    aicl = AICLContextSelector(k_max=3)
    aicl.fit(df, labels)
    result = aicl.transform(df)
    # output rows should never exceed input rows
    assert len(result) <= len(df)


def test_unfitted_raises_error():
    df, _ = make_dummy_data()
    aicl = AICLContextSelector(k_max=3)
    with pytest.raises(Exception):
        aicl.transform(df)


def test_predict_k_returns_dict():
    df, labels = make_dummy_data()
    aicl = AICLContextSelector(k_max=3)
    aicl.fit(df, labels)
    preds = aicl.predict_k(df)
    assert isinstance(preds, dict)
    assert 'q1' in preds
    assert 'q2' in preds


def test_build_labels_utility():
    df, _ = make_dummy_data()
    answers_df = pd.DataFrame({
        'qid': ['q1', 'q2', 'q3'],
        'answers': ['python', 'einstein', 'java']
    })
    labels = AICLContextSelector.build_labels_from_results(
        df, answers_df, answer_col='answers', k_max=3
    )
    assert len(labels) == 3
    assert len(labels[0]) == 3