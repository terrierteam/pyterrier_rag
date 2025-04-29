import pytest
import pandas as pd
from pyterrier_rag._util import (push_columns, push_columns_dict,
                                 find_maximum_push, find_maximum_push_dict)


# Tests for push_queries
def test_push_queries_basic():
    df = pd.DataFrame({'query': ['q1'], 'doc': ['d1']})
    result = push_columns(df)
    expected = pd.DataFrame({'query_0': ['q1'], 'doc': ['d1']})
    pd.testing.assert_frame_equal(result, expected)
    assert 'query' not in result.columns


def test_push_queries_existing_query0():
    df = pd.DataFrame({'query': ['q1'], 'query_0': ['q0'], 'doc': ['d1']})
    result = push_columns(df)
    assert set(result.columns) == {'query_0', 'query_1', 'doc'}
    assert result['query_0'].iloc[0] == 'q1'
    assert result['query_1'].iloc[0] == 'q0'


def test_push_queries_inplace():
    df = pd.DataFrame({'query': ['q1']})
    df = push_columns(df, inplace=True)
    assert 'query_0' in df.columns
    assert 'query' not in df.columns


def test_push_queries_base_column():
    df = pd.DataFrame({'question': ['q1']})
    result = push_columns(df, base_column='question')
    assert 'question_0' in result.columns


def test_push_queries_missing_query():
    df = pd.DataFrame({'doc': ['d1']})
    with pytest.raises(KeyError):
        push_columns(df)


# Tests for push_queries_dict
def test_push_queries_dict_single():
    inp = [{'query': 'q1', 'doc': 'd1'}]
    result = push_columns_dict(inp)
    assert result == [{'query_0': 'q1', 'doc': 'd1'}]


def test_push_queries_dict_keep_original():
    inp = {'query': 'q1'}
    result = push_columns_dict(inp, keep_original=True)
    assert result == {'query_0': 'q1', 'query': 'q1'}


def test_push_queries_dict_existing_query0():
    inp = {'query': 'q1', 'query_0': 'q0'}
    result = push_columns_dict(inp)
    assert result == {'query_0': 'q1', 'query_1': 'q0'}


def test_push_queries_dict_iterable():
    inp = [{'query': 'q1'}, {'query': 'q2'}]
    result = list(push_columns_dict(inp))
    assert result == [{'query_0': 'q1'}, {'query_0': 'q2'}]


# Tests for find_maximum_push
def test_find_maximum_push_basic():
    df = pd.DataFrame({'query_0': [1], 'query_1': [2]})
    max_col, max_val = find_maximum_push(df)
    assert max_col == 'query_1'
    assert max_val == 1


def test_find_maximum_push_single():
    df = pd.DataFrame({'query_0': [1]})
    max_col, max_val = find_maximum_push(df)
    assert max_col == 'query_0'
    assert max_val == 0


def test_find_maximum_push_none():
    df = pd.DataFrame({'doc': [1]})
    max_col, max_val = find_maximum_push(df)
    assert max_col is None
    assert max_val == -1


# Tests for find_maximum_push_dict
def test_find_maximum_push_dict_single():
    inp = {'query_0': 'q1', 'query_1': 'q2'}
    max_col, max_val = find_maximum_push_dict(inp)
    assert max_col == 'query_1'
    assert max_val == 1


def test_find_maximum_push_dict_iterable():
    inp = [{'query_0': 'q1'}, {'query_2': 'q2'}]
    results = list(find_maximum_push_dict(inp))
    assert results == [('query_0', 0), ('query_2', 2)]
