import pandas as pd
import pytest

from pyterrier_rag.prompt import Concatenator, PromptTransformer, prompt
from pyterrier_rag.prompt.wrapper import prompt as wrapper_prompt

pytestmark = pytest.mark.filterwarnings("error::DeprecationWarning")


def test_legacy_prompt_function_warns_and_renders():
    with pytest.warns(DeprecationWarning, match=r"pyterrier_rag\.prompt\.prompt"):
        fn = prompt("Q: {{ query }}")
    assert fn(query="hello") == "Q: hello"


def test_legacy_wrapper_prompt_warns_and_renders():
    with pytest.warns(DeprecationWarning, match=r"pyterrier_rag\.prompt\.prompt"):
        fn = wrapper_prompt("A: {{ answer }}")
    assert fn(answer="42") == "A: 42"


def test_legacy_prompt_transformer_warns_and_transforms():
    with pytest.warns(DeprecationWarning, match=r"pyterrier_rag\.prompt\.PromptTransformer"):
        tf = PromptTransformer(
            instruction="Question: {{ query }}\nContext: {{ qcontext }}",
            input_fields=["query", "qcontext"],
        )

    inp = pd.DataFrame.from_records(
        [{"qid": "q1", "query": "capital?", "qcontext": "Paris is in France."}]
    )
    out = tf.transform(inp)
    assert len(out) == 1
    rec = out.iloc[0].to_dict()
    assert rec["qid"] == "q1"
    assert rec["query_0"] == "capital?"
    assert "Question: capital?" in rec["prompt"]
    assert "Context: Paris is in France." in rec["prompt"]


def test_legacy_concatenator_warns_and_aggregates():
    with pytest.warns(DeprecationWarning, match=r"pyterrier_rag\.prompt\.Concatenator"):
        cat = Concatenator()

    inp = pd.DataFrame.from_records(
        [
            {"qid": "q1", "query": "what", "docno": "d1", "text": "doc one", "score": 0.5},
            {"qid": "q1", "query": "what", "docno": "d2", "text": "doc two", "score": 0.8},
        ]
    )
    out = cat.transform(inp)
    assert len(out) == 1
    rec = out.iloc[0].to_dict()
    assert rec["qid"] == "q1"
    assert rec["query"] == "what"
    assert rec["qcontext"] == "doc two\ndoc one"
