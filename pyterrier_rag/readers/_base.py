from typing import Union

import pandas as pd
import pyterrier as pt
import pyterrier_alpha as pta

from pyterrier_rag.backend import Backend
from pyterrier_rag.prompt import PromptTransformer


GENERIC_PROMPT = (
    "Use the context information to answer the Question: \n Context: {{ qcontext }} \n Question: {{ query }} \n Answer:"
)


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
        from pyterrier_rag.prompt import Concatenator
        from pyterrier_rag.readers import Reader

        flant5 = Reader(Seq2SeqLMBackend('google/flan-t5-base'))
        bm25_flant5 = bm25_ret % 10 >> Concatenator() >> flant5
        bm25_flant5.search("What is the capital of France?")

    Example using a remote LLM::

        from pyterrier_rag.backend import OpenAIBackend
        from pyterrier_rag.prompt import Concatenator
        from pyterrier_rag.readers import Reader

        llamma = Reader(OpenAIBackend("llama-3-8b-instruct", api_key="your_api_key", base_url="your_base_url"))
        bm25_llamma = bm25_ret % 10 >> Concatenator() >> llamma
        bm25_llamma.search("What is the capital of Italy?")


    """
    def __init__(
        self,
        backend: Union[Backend, str],
        prompt: Union[PromptTransformer, str] = GENERIC_PROMPT,
        output_field: str = "qanswer",
    ):
        self.prompt = prompt
        self.backend = backend
        self.output_field = output_field
        self.__post_init__()

    def __post_init__(self):
        if isinstance(self.prompt, str):
            self.prompt = PromptTransformer(
                instruction=self.prompt,
                model_name_or_path=self.backend.model_id,
            )
        if isinstance(self.prompt, PromptTransformer):
            self.prompt.set_output_attribute(self.backend.supports_message_input)
            if self.prompt.expects_logprobs and not self.backend.supports_logprobs:
                raise ValueError("The LLM does not support logprobs")
            elif self.prompt.expects_logprobs and self.backend.supports_logprobs:
                self.backend = self.backend.logprobs_generator()
            else:
                self.backend = self.backend.text_generator()

    def transform_inputs(self):
        return pt.inspect.transformer_inputs(self.prompt)
    
    def transform_outputs(self, inp_cols):
        out = pt.inspect.transformer_outputs(self.prompt, inp_cols)
        return out + [self.output_field]

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        pta.validate.columns(inp, includes=['qid', *self.prompt.input_fields])

        if inp is None or inp.empty:
            return pd.DataFrame(columns=[self.output_field, self.prompt.output_field, "qid"])
        prompts = self.prompt(inp)
        outputs = self.backend(prompts)
        answers = self.prompt.answer_extraction(outputs)

        prompts[self.output_field] = answers[self.output_field]
        return prompts
