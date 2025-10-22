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
        answer_extraction: callable = lambda outputs: outputs,
        output_field: str = "qanswer",
    ):
        self.backend = backend
        self.prompt = prompt if callable(prompt) else jinja_formatter(prompt)
        self.answer_extraction = answer_extraction
        self.output_field = output_field

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        pta.validate.columns(inp, includes=['qid', 'query', 'docno', 'text'])

        if inp is None or inp.empty:
            return pd.DataFrame(columns=["qid", self.output_field, 'qprompt'])

        prompt_frame = pta.DataFrameBuilder(['qid', 'query', 'qprompt'])
        for qid, group in inp.groupby('qid'):
            prompt = self.prompt(docs=group.iterrows(), **group[pt.model.query_columns(inp)].iloc[0])
            prompt_frame.extend({'qid': qid, 'query': group['query'].iloc[0], 'qprompt': prompt})

        output = self.backend(prompt_frame.to_df())
        output[self.output_field] = output[self.backend.output_fields].apply(self.answer_extraction)

        return output
