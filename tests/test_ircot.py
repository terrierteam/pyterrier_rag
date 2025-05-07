import pandas as pd
import pytest
import torch
import pyterrier as pt

from pyterrier_rag.frameworks.ir_cot import IRCOT, ircot_system_message, ircot_prompt, ircot_example_format
from pyterrier_rag.backend import Backend, BackendOutput

# Detailed mock backend based on the Backend spec

class DummyTokenizer:
    def decode(self, output):
        return ''.join(map(str, output))
    def encode(self, text):
        # simple stub: split on whitespace and return lengths
        return [len(w) for w in text.split()]
    
class FakeBackend(Backend):
    _model_name_or_path = 'fake_model'
    _support_logits = True
    _logit_type = None
    _api_type = None
    _remove_prompt = False

    def __init__(self, **kwargs):
        # initialize with default parameters
        super().__init__(**kwargs)
        # add missing tokenizer attribute for context config
        self.tokenizer = DummyTokenizer()

    def generate(self, inp):
        # Generate BackendOutput instances for each input query
        outputs = []
        for _ in inp:
            outputs.append(BackendOutput(text="So the answer is: fake answer", logits=None, prompt_length=0))
        return outputs

class DummyRetriever:
    def __init__(self):
        self.search_calls = []
    def __mod__(self, max_docs):
        self.max_docs = max_docs
        return self
    def search(self, query):
        self.search_calls.append(query)
        # Return a simple DataFrame with required columns
        return pd.DataFrame([{'qid': '1', 'docno': 'd1', 'score': 1.0, 'text': 'dummy text'}])

    def __call__(self, *args, **kwargs):
        return self.search(*args, **kwargs)

class DummyReader:
    def __init__(self, backend, prompt):
        self.backend = backend
        self.prompt = prompt
        self.transform_calls = []
    def transform(self, docs_df):
        self.transform_calls.append(docs_df.copy())
        # Always return a "final" answer to trigger exit_condition
        return pd.DataFrame({'qanswer': ["So the answer is: dummy answer"]})
    def __call__(self, df):
        return self.transform(df)

# Simple conversation template stub for PromptTransformer mocks
class SimpleConvTemplate:
    def __init__(self):
        self.messages = []
    def copy(self):
        # Return a shallow copy
        new = SimpleConvTemplate()
        new.messages = list(self.messages)
        return new
    def set_system_message(self, system_message):
        # Prepend system message
        self.messages.insert(0, {'role': 'system', 'content': system_message})
    def append_message(self, role, content):
        self.messages.append({'role': role, 'content': content})
    def get_prompt(self):
        # Join all messages into a single prompt string
        return "".join(f"{m['role']}: {m['content']}" for m in self.messages)

# Test iteration threshold logic
@pytest.fixture(autouse=True)
def patch_transformers_and_reader(monkeypatch):
    # Fake prompt and context transformers
    class FakePromptTransformer(pt.Transformer):
        expect_logits=False
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        def set_output_attribute(self, attr):
            pass
        def transform(self, inp):
            return inp
    class FakeConcatenator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        def __rshift__(self, other):
            return other
    # Patch transformers
    monkeypatch.setattr('pyterrier_rag.prompt.PromptTransformer', FakePromptTransformer)
    monkeypatch.setattr('pyterrier_rag.prompt.Concatenator', FakeConcatenator)
    # Patch Reader
    monkeypatch.setattr('pyterrier_rag.readers.Reader', DummyReader)
    yield


def test_exceeded_max_iterations():
    backend = FakeBackend()
    retriever = DummyRetriever()
    ircot = IRCOT(retriever=retriever, backend=backend, max_iterations=2)
    assert not ircot._exceeded_max_iterations(1)
    assert ircot._exceeded_max_iterations(2)
    assert ircot._exceeded_max_iterations(3)


def test_make_default_configs():
    backend = FakeBackend()
    retriever = DummyRetriever()
    ircot = IRCOT(retriever=retriever, backend=backend)
    # Prompt config
    prompt_cfg = ircot._make_default_prompt_config()
    assert prompt_cfg['model_name_or_path'] == backend.model_name_or_path
    assert prompt_cfg['system_message'] == ircot_system_message
    assert prompt_cfg['instruction'] == ircot_prompt
    assert prompt_cfg['output_field'] == 'qanswer'
    assert prompt_cfg['input_fields'] == ['query', 'qcontext', 'prev']
    # Context config
    context_cfg = ircot._make_default_context_config()
    assert context_cfg['in_fields'] == ['text']
    assert context_cfg['out_field'] == 'qcontext'
    assert context_cfg['tokenizer'] == backend.tokenizer
    assert context_cfg['max_length'] == backend.max_input_length
    assert context_cfg['max_elements'] == ircot.max_docs
    assert 'intermediate_format' in context_cfg


def test_init_with_provided_transformers():
    from pyterrier_rag.prompt import PromptTransformer, Concatenator
    backend = FakeBackend()
    retriever = DummyRetriever()
    fake_ctx = Concatenator(dummy='x')
    fake_prompt = PromptTransformer(dummy='y')
    ircot = IRCOT(retriever=retriever, backend=backend,
                  prompt=fake_prompt, context_aggregation=fake_ctx)
    assert ircot.context_aggregation is fake_ctx
    assert ircot.prompt is fake_prompt


def test_transform_by_query_with_exit_condition():
    backend = FakeBackend()
    retriever = DummyRetriever()
    ircot = IRCOT(retriever=retriever, backend=backend)
    # Ensure DummyReader is used
    ircot.reader = DummyReader(backend, ircot.prompt)
    # Exit immediately
    ircot.exit_condition = lambda output: True
    result = ircot.transform_by_query([{'qid': 'q1', 'query': 'test'}])
    assert isinstance(result, list) and len(result) == 1
    rec = result[0]
    assert rec['qid'] == 'q1'
    assert rec['query'] == 'test'
    assert rec['qanswer'] == 'So the answer is: dummy answer'
    # Only initial search should have been called
    assert retriever.search_calls == ['test']


def test_transform_by_query_max_iterations():
    backend = FakeBackend()
    retriever = DummyRetriever()
    ircot = IRCOT(retriever=retriever, backend=backend, max_iterations=1)
    # Reader that never triggers exit
    class ReaderNoExit:
        def __init__(self, backend, prompt): pass
        def transform(self, docs_df):
            return pd.DataFrame({'qanswer': ['intermediate']})
        def __call__(self, df):
            return self.transform(df)
    ircot.reader = ReaderNoExit(None, None)
    ircot.exit_condition = lambda output: False
    result = ircot.transform_by_query([{'qid': 'q2', 'query': 'init'}])
    assert result[0]['qanswer'] == 'intermediate'
    # Only initial search
    assert retriever.search_calls == ['init']


def test_transform_by_query_query_mismatch_error():
    backend = FakeBackend()
    retriever = DummyRetriever()
    ircot = IRCOT(retriever=retriever, backend=backend)
    ircot.reader = DummyReader(backend, ircot.prompt)
    # Provide two different queries should raise assertion
    with pytest.raises(AssertionError):
        list(ircot.transform_by_query([
            {'qid': 'x', 'query': 'one'},
            {'qid': 'x', 'query': 'two'}
        ]))
