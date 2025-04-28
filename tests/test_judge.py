import pandas as pd
import pytest

from pyterrier_rag.frameworks.llm_as_judge import llmjudge_fn

# Fixtures to mock backend and prompt transformer
class FakeBackend:
    model_name_or_path: str = 'gpt-4o-mini'
    def __init__(self):
        self.generate_calls = []
    def generate(self, prompts):
        # record prompts and return a consistent rating string per prompt
        self.generate_calls.append(prompts)
        # Return a list of strings where the first token is '3'
        return ["3 good"] * len(prompts)

@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    # Monkeypatch get_backend to return our fake backend
    monkeypatch.setattr('pyterrier_rag.frameworks.llm_as_judge.get_backend', lambda backend_type, model_name: FakeBackend())
    # Monkeypatch PromptTransformer to our fake
    yield

# Helper to build DataFrames
def make_qrels(texts, relevances, query_id='q1'):
    return pd.DataFrame({
        'query_id': [query_id] * len(texts),
        'relevance': relevances,
        'text': texts,
    })

def make_res(answer, query_id='q1'):
    return pd.DataFrame({'query_id': [query_id], 'qanswer': [answer]})

# Test empty res raises assertion
def test_empty_res_raises():
    qrels = make_qrels(['ref'], [3])
    res = pd.DataFrame(columns=['query_id', 'qanswer'])
    with pytest.raises(AssertionError):
        llmjudge_fn(qrels, res, backend_type='any', model_name='model')

# Test empty qrels raises assertion
def test_empty_qrels_raises():
    qrels = pd.DataFrame(columns=['query_id', 'relevance', 'text'])
    res = make_res('ans')
    with pytest.raises(AssertionError):
        llmjudge_fn(qrels, res, backend_type='any', model_name='model')

# Test no qrels above threshold
def test_no_qrels_above_rel_raises():
    qrels = make_qrels(['ref1', 'ref2'], [1, 2])  # below default rel=3
    res = make_res('ans')
    with pytest.raises(AssertionError):
        llmjudge_fn(qrels, res, backend_type='any', model_name='model')

# Parametrized tests for aggregation methods
@pytest.mark.parametrize("agg, expected", [
    ('max', 3),
    ('min', 3),
    ('sum', 6),
    ('avg', 3.0),
    ('none', [3, 3]),
])
def test_aggregation_methods(agg, expected):
    # Two relevant qrels => duplication of single prediction to length 2
    qrels = make_qrels(['r1', 'r2'], [3, 5])
    res = make_res('answer')
    score = llmjudge_fn(qrels, res, backend_type='openai', model_name='gpt-4o-mini', rel=1, agg=agg)
    assert score == expected

# Test invalid aggregation raises ValueError
def test_invalid_aggregation():
    qrels = make_qrels(['r1'], [3])
    res = make_res('answer')
    with pytest.raises(ValueError):
        llmjudge_fn(qrels, res, backend_type='openai', model_name='gpt-4o-mini', agg='unknown')

# Test that generate is called with prompts matching the fake PromptTransformer
def test_prompt_and_generate_calls():
    qrels = make_qrels(['ref1', 'ref2'], [3, 4])
    res = make_res('pred')
    # Call with agg='none' to inspect raw outputs
    result = llmjudge_fn(qrels, res, backend_type='openai', model_name='gpt-4o-mini', rel=1, agg='none')
    # The global backend_obj in module should now be our FakeBackend
    from pyterrier_rag.frameworks.llm_as_judge import backend_obj, prompt_obj
    assert isinstance(backend_obj, FakeBackend)
    # FakeBackend.generate should have been called once
    prompts_sent = backend_obj.generate_calls[0]
    # There should be two prompts, one per reference
    assert len(prompts_sent) == 2