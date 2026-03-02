import pyterrier as pt
import pandas as pd
import pytest
from typing import Iterable
from pyterrier_rag.backend import Backend, BackendOutput
from pyterrier_rag.readers._base import Reader


class RecordingBackend(Backend):
    """Backend stub that records raw inputs passed into generate()."""

    supports_logprobs = True
    supports_message_input = False

    def __init__(self, **kwargs):
        super().__init__("dummy-model", **kwargs)
        self.seen_prompts = []

    def generate(
        self,
        inp,
        return_logprobs=False,
        max_new_tokens=None,
        num_responses=1,
        stop_sequences=None,
    ) -> Iterable[BackendOutput]:
        self.seen_prompts.extend(inp)
        outputs = []
        for prompt in inp:
            for _ in range(num_responses):
                outputs.append(
                    BackendOutput(
                        text=f"resp:{prompt}",
                        logprobs=[{'a': 1.0}],
                    )
                )
        return outputs


class TestReader:
    @staticmethod
    def _input_df():
        return pd.DataFrame(
            [
                {"qid": "q1", "query": "What is Paris?", "docno": "d1", "text": "Paris is in France."},
                {"qid": "q1", "query": "What is Paris?", "docno": "d2", "text": "It is the capital city."},
                {"qid": "q2", "query": "What is Rome?", "docno": "d3", "text": "Rome is in Italy."},
            ]
        )

    def test_reader_with_jinja_string_prompt(self):
        backend = RecordingBackend()
        reader = Reader(
            backend=backend,
            prompt="Q={{ query }} | {% for _, d in docs %}{{ d.text }} {% endfor %}",
            output_field="answer",
            answer_extraction=lambda x: x,
        )

        out = reader.transform(self._input_df())
        assert len(out) == 2
        assert set(out["qid"]) == {"q1", "q2"}
        assert "answer" in out.columns
        assert "Paris is in France." in backend.seen_prompts[0]
        assert "It is the capital city." in backend.seen_prompts[0]
        assert "What is Rome?" in backend.seen_prompts[1]

    def test_reader_with_callable_string_prompt(self):
        backend = RecordingBackend()

        def prompt_fn(docs, query, **kwargs):
            context = " | ".join(d.text for _, d in docs)
            return f"Question={query}; Context={context}"

        reader = Reader(
            backend=backend,
            prompt=prompt_fn,
            output_field="answer",
            answer_extraction=lambda x: x,
        )
        out = reader.transform(self._input_df())

        assert len(out) == 2
        assert "Question=What is Paris?" in backend.seen_prompts[0]
        assert "Paris is in France." in backend.seen_prompts[0]
        assert "It is the capital city." in backend.seen_prompts[0]

    def test_reader_with_string_prompt_on_message_backend_adds_system_message(self):
        class MessageBackend(RecordingBackend):
            supports_message_input = True

        backend = MessageBackend()
        reader = Reader(
            backend=backend,
            prompt="Question: {{ query }}",
            system_prompt="Use only provided context.",
            output_field="answer",
            answer_extraction=lambda x: x,
        )
        reader.transform(self._input_df())

        prompt = backend.seen_prompts[0]
        assert isinstance(prompt, list)
        assert prompt[0] == {"role": "system", "content": "Use only provided context."}
        assert prompt[1]["role"] == "user"
        assert "Question: What is Paris?" in prompt[1]["content"]

    def test_reader_with_callable_message_prompt_on_message_backend(self):
        class MessageBackend(RecordingBackend):
            supports_message_input = True

        backend = MessageBackend()

        def prompt_fn(docs, query, **kwargs):
            context = "\n".join(d.text for _, d in docs)
            return [{"role": "user", "content": f"{query}\n{context}"}]

        reader = Reader(
            backend=backend,
            prompt=prompt_fn,
            system_prompt="System rule.",
            output_field="answer",
            answer_extraction=lambda x: x,
        )
        reader.transform(self._input_df())

        prompt = backend.seen_prompts[0]
        assert prompt[0] == {"role": "system", "content": "System rule."}
        assert prompt[1]["role"] == "user"
        assert "What is Paris?" in prompt[1]["content"]

    def test_reader_with_callable_message_prompt_on_text_backend_flattens_content(self):
        backend = RecordingBackend()

        def prompt_fn(docs, query, **kwargs):
            _ = list(docs)
            return [
                {"role": "system", "content": "Inner system"},
                {"role": "user", "content": f"Question: {query}"},
            ]

        reader = Reader(
            backend=backend,
            prompt=prompt_fn,
            system_prompt="Outer system",
            output_field="answer",
            answer_extraction=lambda x: x,
        )
        reader.transform(self._input_df())

        flattened = backend.seen_prompts[0]
        assert isinstance(flattened, str)
        assert flattened.startswith("Outer system")
        assert "Inner system" in flattened
        assert "Question: What is Paris?" in flattened

    def test_reader_requires_qid_and_query_columns(self):
        backend = RecordingBackend()
        reader = Reader(backend=backend)
        with pytest.raises(Exception):
            reader.transform(pd.DataFrame([{"qid": "q1", "text": "x"}]))

    @pt.testing.transformer_test_class
    def test_reader():
        x = Reader(backend=RecordingBackend())
        x.transform(pd.DataFrame(columns=['qid', 'query']))
        return x
