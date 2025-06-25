from typing import Union
import pandas as pd
import pyterrier as pt

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

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        prompts = self.prompt(inp)
        outputs = self.backend(prompts)
        answers = self.prompt.answer_extraction(outputs)

        prompts[self.output_field] = answers[self.output_field]
        return prompts
