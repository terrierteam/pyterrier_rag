import pytest
import pandas as pd
import pyterrier as pt

from pyterrier_rag.provence import Provence



@pt.testing.transformer_test_class
def test_provence():
    return Provence()


def test_provence_simple():
    try:
        model = Provence.provence(batch_size=1, remove_empty=False)
    except Exception as exc:
        pytest.skip(f"Provence model unavailable in this environment: {exc}")

    inp = pd.DataFrame([
        {
            "qid": "q1",
            "query": "What is the capital of France?",
            "docno": "d1",
            "title": "France",
            "text": "France is a country in Europe. Paris is its capital city.",
        }
    ])
    out = model(inp)

    assert len(out) == 1
    assert "text" in out.columns
    assert "score" in out.columns
    assert "rank" in out.columns
