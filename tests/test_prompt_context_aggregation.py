import pytest
from pyterrier_rag.prompt._context_aggregation import concat, dataframe_concat, ContextAggregationTransformer
import pandas as pd


class MockTokenizer:
    def encode(self, text):
        return text.split()  # Split by whitespace
    def decode(self, tokens):
        return ' '.join(tokens)


class MockCharTokenizer:
    def encode(self, text):
        return list(text)  # Each character is a token
    def decode(self, tokens):
        return ''.join(tokens)


def test_concat_basic():
    input_texts = ["a", "b", "c"]
    assert concat(input_texts) == "a\nb\nc"


def test_concat_intermediate_format():
    result = concat(["a", "b"], intermediate_format=lambda x: f"Doc: {x}")
    assert result == "Doc: a\nDoc: b"


def test_concat_with_tokenizer_truncation():
    tokenizer = MockCharTokenizer()
    input_texts = ["abcd", "efgh"]
    result = concat(
        input_texts,
        tokenizer=tokenizer,
        max_length=8,
        max_per_context=4,
        truncation_rate=1,
    )
    assert result == "abc\nefg\n"


def test_concat_max_elements():
    input_texts = ["a", "b", "c", "d"]
    result = concat(input_texts, max_elements=2)
    assert result == "a\nb"


def test_concat_empty_input():
    assert concat([]) == ""


def test_dataframe_concat_basic():
    df = pd.DataFrame({
        "text": ["a", "b", "c"]
    })
    result = dataframe_concat(df, relevant_fields=["text"])
    assert result == "a\nb\nc"


def test_dataframe_concat_by_query():
    df = pd.DataFrame({
        "query_id": [1, 1, 2, 2],
        "query": ["q1", "q1", "q2", "q2"],
        "text": ["a", "b", "c", "d"]
    })
    result = dataframe_concat(df, by_query=True)
    assert len(result) == 2
    assert result[result['query_id'] == 1]['context'].iloc[0] == "a\nb"
    assert result[result['query_id'] == 2]['context'].iloc[0] == "c\nd"


def test_dataframe_concat_intermediate_format():
    df = pd.DataFrame({
        "text": ["a", "b"]
    })
    format_func = lambda x: f"Text: {x['text']}"
    result = dataframe_concat(df, intermediate_format=format_func)
    assert result == "Text: a\nText: b"


def test_dataframe_concat_empty():
    df = pd.DataFrame(columns=["text"])
    assert dataframe_concat(df) == ""


def test_context_aggregation_transformer_default():
    df = pd.DataFrame({
        "query_id": [1, 1, 2, 2],
        "query": ["q1", "q1", "q2", "q2"],
        "text": ["a", "b", "c", "d"]
    })
    transformer = ContextAggregationTransformer(per_query=True)
    transformed_df = transformer.transform(df)
    assert len(transformed_df) == 2
    assert transformed_df[transformed_df['query_id'] == 1]['context'].iloc[0] == "a\nb"
    assert transformed_df[transformed_df['query_id'] == 2]['context'].iloc[0] == "c\nd"


def test_context_aggregation_transformer_custom_agg():
    df = pd.DataFrame({
        "query_id": [1, 1],
        "text": ["a", "b"]
    })
    transformer = ContextAggregationTransformer(
        aggregate_func=lambda x: ", ".join(x['text']),
        per_query=True
    )
    transformed_df = transformer.transform(df)
    assert transformed_df['context'].iloc[0] == "a, b"


def test_context_aggregation_transformer_with_tokenizer():
    df = pd.DataFrame({
        "query_id": [1, 1],
        "text": ["abcd", "efgh"]
    })
    tokenizer = MockCharTokenizer()
    transformer = ContextAggregationTransformer(
        tokenizer=tokenizer,
        max_length=8,
        max_per_context=4,
        truncation_rate=1,
        per_query=True
    )
    transformed_df = transformer.transform(df)
    assert transformed_df['context'].iloc[0] == "abc\nefg\n"
