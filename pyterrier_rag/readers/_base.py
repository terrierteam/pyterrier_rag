from typing import Union
import pandas as pd
import pyterrier as pt
import torch
from typing import Union, Optional

from pyterrier_rag.backend import Backend
from pyterrier_rag.prompt import PromptTransformer



class Reader(pt.Transformer):
    def __init__(
        self,
        backend: Union[Backend, str],
        prompt: Union[PromptTransformer, str] = None,
        context_aggregation: Optional[callable] = None,
        output_field: str = "qanswer",
    ):
        self.prompt = prompt
        self.backend = backend
        self.context_aggregation = context_aggregation
        self.output_field = output_field
        self.__post_init__()

    def __post_init__(self):
        if isinstance(self.prompt, str):
            self.prompt = PromptTransformer(
                instruction=self.prompt,
                model_name_or_path=self.backend._model_name_or_path,
            )

        self.prompt.set_output_attribute(self.backend._api_type)
        if self.prompt.expect_logits and not self.backend._support_logits:
            raise ValueError("The LLM does not support logits")
        elif self.prompt.expect_logits and self.backend._support_logits:
            self.backend = self.backend.logit_generator()
        else:
            self.backend = self.backend.text_generator()

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        prompts = self.prompt(inp)
        outputs = self.backend(prompts)
        answers = self.prompt.answer_extraction(outputs)

        prompts[self.output_field] = answers
        return prompts
