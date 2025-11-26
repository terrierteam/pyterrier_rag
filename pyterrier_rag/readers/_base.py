from typing import Union


import pandas as pd
import pyterrier as pt
import pyterrier_alpha as pta
import pyterrier_alpha as pta

from pyterrier_rag.backend import Backend
from pyterrier_rag.prompt.default import DefaultPrompt
from pyterrier_rag.prompt.jinja import jinja_formatter


class Reader(pt.Transformer):
    """
    Transformer that generates answers from context and queries using an LLM backend.

    Combines a PromptTransformer with a Backend to produce text or logprobs,
    then applies answer extraction to return final responses.

    Parameters:
        backend (Backend or str): A Backend instance or model identifier string.
        prompt (PromptTransformer or str): Prompt template or raw instruction.
        output_field (str): Field name in the output DataFrame for answers.

    Raises:
        ValueError: If the prompt expects logprobs but the backend does not support logprobs.

    Example using a local LLM::

        from pyterrier_rag.backend import Seq2SeqLMBackend
        from pyterrier_rag.readers import Reader

        flant5 = Reader(Seq2SeqLMBackend('google/flan-t5-base'))
        bm25_flant5 = bm25_ret % 10 >> flant5
        bm25_flant5.search("What is the capital of France?")

    Example using a remote LLM::

        from pyterrier_rag.backend import OpenAIBackend
        from pyterrier_rag.readers import Reader

        llamma = Reader(OpenAIBackend("llama-3-8b-instruct", api_key="your_api_key", base_url="your_base_url"). 
        bm25_llamma = bm25_ret % 10 >> llamma
        bm25_llamma.search("What is the capital of Italy?")


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

        prompt_frame = pta.DataFrameBuilder(['qid', 'query', 'prompt'])
        for qid, group in inp.groupby('qid'):
            prompt = self.make_prompt_from(docs=group.iterrows(), **group[pt.model.query_columns(inp)].iloc[0])
            prompt_frame.extend({'qid': qid, 'query': group['query'].iloc[0], 'prompt': prompt})

        output = self.backend(prompt_frame.to_df())
        output[self.output_field] = output['qanswer'].apply(self.answer_extraction)

        return output
