import pytest
import pandas as pd
import pyterrier as pt

from pyterrier_rag.provence import Provence


def test_inferred_inputs():
    try:
        model = Provence.provence(batch_size=1, device_map="cpu")
    except Exception as exc:
        pytest.skip(f"Provence model unavailable in this environment: {exc}")

    inputs = pt.inspect.transformer_inputs(model)
    assert ["qid", "query", "docno", "text"] in inputs
    assert ["qid", "query", "docno", "title", "text"] in inputs


@pt.testing.transformer_test_class
def test_inspect():
    try:
        return Provence.provence(batch_size=1, device_map="cpu")
    except Exception as exc:
        pytest.skip(f"Provence model unavailable in this environment: {exc}")


def test_provence_real_model_smoke():
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
    assert out.iloc[0]["text_0"] == inp.iloc[0]["text"]
    assert "text" in out.columns
    assert "score" in out.columns
    assert "rank" in out.columns
