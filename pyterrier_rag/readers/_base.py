from typing import Union


import pandas as pd
import pyterrier as pt
import pyterrier_alpha as pta

from pyterrier_rag.backend import Backend
from pyterrier_rag.prompt.default import DefaultPrompt
from pyterrier_rag.prompt.jinja import jinja_formatter


class Reader(pt.Transformer):
    """
    Transformer that generates answers from context and queries using an LLM backend.

    For each ``qid`` group, Reader builds one prompt from the retrieved documents and
    query columns, sends it to the backend, and writes the answer to ``output_field``.

    Prompt contract in this refactored API:
    - ``prompt`` can be a jinja ``str`` template or a callable.
    - String prompts are rendered with ``jinja_formatter`` and receive
      ``docs=<group.iterrows()>`` plus all query columns (for example ``query``).
    - Callable prompts must accept ``docs`` and query columns via ``**kwargs`` and
      return either a string prompt or a list of chat messages.

    Parameters:
        backend (Backend or str): A Backend instance or model identifier string.
        prompt (callable or str): Prompt function or jinja template.
        system_prompt (str, optional): Prepended for text backends, or added as a
            ``system`` message for chat backends.
        answer_extraction (callable): Maps backend ``qanswer`` values to final output.
        output_field (str): Field name in the output DataFrame for answers.

    Example with a jinja template::

        from pyterrier_rag.backend import Seq2SeqLMBackend
        from pyterrier_rag.readers import Reader

        reader = Reader(
            backend=Seq2SeqLMBackend("google/flan-t5-base"),
            prompt="Question: {{ query }}\nContext:{% for _, d in docs %}\n{{ d.text }}{% endfor %}\nAnswer:",
        )

    Example with a callable prompt for chat backends::

        from pyterrier_rag.backend import OpenAIBackend
        from pyterrier_rag.readers import Reader

        def chat_prompt(docs, query, **kwargs):
            context = "\n".join(d.text for _, d in docs)
            return [{"role": "user", "content": f"Question: {query}\nContext: {context}\nAnswer:"}]

        reader = Reader(
            backend=OpenAIBackend("gpt-4o-mini", api_key="..."),
            prompt=chat_prompt,
            system_prompt="Answer using only the provided context.",
        )

    """
    def __init__(
        self,
        backend: Union[Backend, str],
        prompt: Union[callable, str] = DefaultPrompt,
        system_prompt: str = None,
        answer_extraction: callable = lambda outputs: outputs,
        output_field: str = "qanswer",
    ):
        self.backend = backend
        self.prompt = prompt if callable(prompt) else jinja_formatter(prompt)
        self.make_prompt_from = (
            self.callable_prompt
            if callable(prompt)
            else self.string_prompt
        )
        self.system_prompt = system_prompt
        self.answer_extraction = answer_extraction
        self.output_field = output_field

    def string_prompt(self, docs, **query_columns):
        prompt_text = self.prompt(docs=docs, **query_columns)
        if self.backend.supports_message_input:
            messages = []
            if self.system_prompt is not None:
                messages.append({'role': 'system', 'content': self.system_prompt})
            messages.append({'role': 'user', 'content': prompt_text})
            return messages
        else:
            if self.system_prompt is not None:
                prompt_text = self.system_prompt + "\n\n" + prompt_text
            return prompt_text

    def callable_prompt(self, docs, **query_columns):
        prompt_output = self.prompt(docs=docs, **query_columns)
        if self.backend.supports_message_input:
            messages = []
            if self.system_prompt is not None:
                messages.append({'role': 'system', 'content': self.system_prompt})
            if isinstance(prompt_output, str):
                messages.append({'role': 'user', 'content': prompt_output})
            else:
                messages.extend(prompt_output)
            return messages
        else:
            if isinstance(prompt_output, str):
                if self.system_prompt is not None:
                    return self.system_prompt + "\n\n" + prompt_output
                return prompt_output
            else:
                # For callable prompts that return messages, extract content
                content = ""
                for msg in prompt_output:
                    if msg.get('role') == 'system':
                        content += msg.get('content', '') + "\n\n"
                    else:
                        content += msg.get('content', '')
                if self.system_prompt is not None:
                    content = self.system_prompt + "\n\n" + content
                return content

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        # Require at least qid and query
        pta.validate.columns(inp, includes=['qid', 'query'])

        if inp is None or inp.empty:
            return pd.DataFrame(columns=["qid", self.output_field, 'prompt'])

        prompt_frame = []
        for qid, group in inp.groupby('qid'):
            prompt = self.make_prompt_from(docs=group.iterrows(), **group[pt.model.query_columns(inp)].iloc[0])
            prompt_frame.append({'qid': qid, 'query': group['query'].iloc[0], 'prompt': prompt})
        output = self.backend(pd.DataFrame(prompt_frame))
        output[self.output_field] = output['qanswer'].apply(self.answer_extraction)

        return output
