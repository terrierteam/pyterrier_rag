import pytest
import pandas as pd
import itertools

import pyterrier_alpha as pta

from pyterrier_rag._util import (
    push_columns,
    push_columns_dict,
    find_maximum_push,
    find_maximum_push_dict,
    intermediate_formatting,
    concat,
    dataframe_concat,
)


class DummyTokenizer:
    """A tokenizer where each character is a token."""
    def encode(self, text: str):
        return list(text)
    def decode(self, tokens):
        return "".join(tokens)


def test_push_columns_basic():
    df = pd.DataFrame({'query': ['a', 'b'], 'other': [1, 2]})
    out = push_columns(df)
    assert 'query_0' in out.columns
    assert list(out['query_0']) == ['a', 'b']
    # original 'query' gone
    assert 'query' not in out.columns


def test_push_columns_keep_original():
    df = pd.DataFrame({'query': ['x']})
    out = push_columns(df, keep_original=True)
    assert 'query_0' in out.columns
    assert 'query' in out.columns
    assert out.at[0, 'query'] == out.at[0, 'query_0']


def test_push_columns_missing_column():
    df = pd.DataFrame({'foo': [1]})
    with pytest.raises(KeyError):
        push_columns(df)


def test_push_columns_dict_single():
    d = {'query': 'a', 'foo': 2}
    out = push_columns_dict(d)
    assert out == {'query_0': 'a', 'foo': 2}


def test_push_columns_dict_list_keep():
    data = [{'query': 'a'}, {'query': 'b'}]
    out = push_columns_dict(data, keep_original=True)
    assert isinstance(out, list)
    assert all('query' in elem and 'query_0' in elem for elem in out)


def test_push_columns_dict_missing():
    with pytest.raises(KeyError):
        push_columns_dict({'foo': 1})


def test_find_maximum_push():
    df = pd.DataFrame({'query_0': [1], 'query_2': [2], 'query_1': [3]})
    col, val = find_maximum_push(df)
    assert col == 'query_2'
    assert val == 2


def test_find_maximum_push_dict_single():
    d = {'query_0': 'a', 'query_3': 'b'}
    col, val = find_maximum_push_dict(d)
    assert col == 'query_3'
    assert val == 3


def test_find_maximum_push_dict_iterable():
    data = [{'query_0': 1}, {'query_1': 2, 'query_5': 3}]
    results = list(find_maximum_push_dict(data))
    assert results == [('query_0', 0), ('query_5', 5)]


def test_intermediate_formatting_no_formatter_dict():
    assert intermediate_formatting({'text': 'hello'}) == 'hello'
    assert intermediate_formatting({'foo': 'bar'}) == ''


def test_intermediate_formatting_no_formatter_str():
    assert intermediate_formatting('x') == 'x'


def test_intermediate_formatting_with_formatter():
    # string → formatted
    fmt1 = lambda x: f"<>{x}<>"
    assert intermediate_formatting('a', fmt1) == '<>a<>'
    # tuple/list → positional unpack
    fmt2 = lambda x, y: x + y
    assert intermediate_formatting(('a', 'b'), fmt2) == 'ab'
    assert intermediate_formatting(['x', 'y'], fmt2) == 'xy'
    # dict → keyword unpack
    fmt3 = lambda k: k
    assert intermediate_formatting({'k': 'v'}, fmt3) == 'v'


def test_concat_no_tokenizer():
    texts = ['foo', 'bar', 'baz']
    assert concat(texts) == 'foo\nbar\nbaz'


def test_concat_with_intermediate_and_max_elements():
    texts = [1, 2, 3, 4]
    fmt = lambda x: str(x * 10)
    out = concat(texts, intermediate_format=fmt, max_elements=2)
    assert out == '10\n20'


def test_concat_with_tokenizer_truncation_and_max_length():
    texts = ['aaaa', 'bbbb', 'cccc']
    tok = DummyTokenizer()
    # initial max_per_qcontext=3 → segments 'aaa','bbb','ccc', combined length=11 > max_length=6
    # then max_per_qcontext→2 → 'aa','bb','cc', combined length=8 > 6
    # then →1 → 'a','b','c', combined length=5 ≤ 6
    out = concat(
        texts,
        tokenizer=tok,
        max_length=6,
        max_per_context=3,
        truncation_rate=1
    )
    assert out == 'a\nb\nc'


def test_dataframe_concat_no_tokenizer():
    df = pd.DataFrame({'text': ['x', 'y']})
    out = dataframe_concat(df)
    assert out == 'x\ny'


def test_dataframe_concat_by_query(monkeypatch):
    df = pd.DataFrame({
        'qid': [1, 1, 2],
        'query': ['q1', 'q1', 'q2'],
        'text': ['a', 'b', 'c']
    })

    # stub out pta.DataFrameBuilder
    class DummyBuilder:
        def __init__(self, cols):
            self.rows = []
        def extend(self, d):
            for i in range(len(d['qid'])):
                self.rows.append({
                    'qid': d['qid'][i],
                    'query': d['query'][i],
                    'qcontext': d['qcontext'][i]
                })
        def to_df(self):
            return pd.DataFrame(self.rows)

    monkeypatch.setattr(pta, 'DataFrameBuilder', DummyBuilder)

    out_df = dataframe_concat(df, by_query=True)
    # one row per qid
    assert set(out_df['qid']) == {1, 2}
    # qcontexts as expected
    ctx = dict(zip(out_df['qid'], out_df['qcontext']))
    assert ctx[1] == 'a\nb'
    assert ctx[2] == 'c'
