import pytest
import pandas as pd

from pyterrier_rag.frameworks.ir_cot import IRCOT


class DummyBackend:
    def __init__(self):
        self.model_name_or_path = "dummy-model"
        self.tokenizer = lambda x: x.split()
        self.max_input_length = 512


class DummyRetriever:
    """Supports `retriever % max_docs` and `search(query)` → DataFrame."""
    def __init__(self):
        self.max_docs = None
        self.queries = []

    def __mod__(self, max_docs: int):
        self.max_docs = max_docs
        return self

    def search(self, query: str) -> pd.DataFrame:
        # return one doc per call, with score = 1.0
        self.queries.append(query)
        return pd.DataFrame({
            "docno": [f"{query}-doc"],
            "score": [1.0]
        })


class DummyReader:
    """Yields a sequence of DataFrames on `transform()` calls."""
    def __init__(self, backend, prompt, outputs):
        self.outputs = outputs.copy()
        self.backend = backend
        self.prompt = prompt

    def transform(self, docs_df: pd.DataFrame) -> pd.DataFrame:
        # pop next pre‑built output
        return self.outputs.pop(0)


@pytest.fixture(autouse=True)
def patch_reader(monkeypatch):
    """Patch Reader → DummyReader to control outputs."""
    from pyterrier_rag.readers import Reader  # adjust to actual import
    def factory(backend, prompt):
        # will be overridden per‑test by setting Reader.outputs_list
        return DummyReader(backend, prompt, factory.outputs_list)
    factory.outputs_list = []
    monkeypatch.setattr("pyterrier_rag.readers.Reader", factory)
    return factory

def test_transform_by_query_exit_condition_immediate(patch_reader):
    # exit_condition returns True on first output
    retr = DummyRetriever()
    backend = DummyBackend()
    # set reader to emit a single DataFrame whose qanswer contains the trigger
    df = pd.DataFrame({"qanswer": ["So the answer is: yay"]})
    patch_reader.outputs_list[:] = [df]

    ir = IRCOT(
        retriever=retr,
        backend=backend,
        max_iterations=5,
        exit_condition=lambda out: out["qanswer"].iloc[0].startswith("So the answer is")
    )

    inp = [{"qid": "Q1", "query": "foo"}]
    out = ir.transform_by_query(inp)
    assert isinstance(out, list) and len(out) == 1
    row = out[0]
    assert row["qid"] == "Q1"
    assert row["query"] == "foo"
    assert row["qanswer"] == "So the answer is: yay"

    # retriever.search was called exactly once for the initial query
    assert retr.queries == ["foo"]


def test_transform_by_query_limited_by_max_iterations(patch_reader):
    # exit_condition never true, but max_iterations=1 → only one pass
    retr = DummyRetriever()
    backend = DummyBackend()
    # reader will emit two different DataFrames if called twice
    df1 = pd.DataFrame({"qanswer": ["step1"]})
    df2 = pd.DataFrame({"qanswer": ["step2"]})
    patch_reader.outputs_list[:] = [df1, df2]

    ir = IRCOT(
        retriever=retr,
        backend=backend,
        max_iterations=1,
        exit_condition=lambda out: False
    )

    inp = [{"qid": "Q2", "query": "bar"}]
    out = ir.transform_by_query(inp)
    assert out[0]["qanswer"] == "step1"
    # retriever.search only for the initial query; no second retrieval
    assert retr.queries == ["bar"]


def test_transform_by_query_asserts_on_mixed_queries():
    retr = DummyRetriever()
    backend = DummyBackend()
    from pyterrier_rag.readers import Reader  # noqa: F401
    # we don't need to set patch_reader.outputs_list here because exit happens after assertion
    ir = IRCOT(retriever=retr, backend=backend)

    # two rows with different 'query' should trigger AssertionError
    inp = [
        {"qid": "X", "query": "alpha"},
        {"qid": "X", "query": "beta"},
    ]
    with pytest.raises(AssertionError):
        ir.transform_by_query(inp)
