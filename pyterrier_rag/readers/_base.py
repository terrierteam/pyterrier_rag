from typing import Union
import pandas as pd
import pyterrier as pt
import torch
from typing import Union, Optional

from pyterrier_rag.backend import Backend
from pyterrier_rag.prompt import (
    PromptTransformer,
    PromptConfig,
    ContextConfig,
    ContextAggregationTransformer,
)


class Reader(pt.Transformer):
    def __init__(
        self,
        backend: Union[Backend, str],
        prompt: Union[PromptTransformer, str] = None,
        context_aggregation: Optional[callable] = None,
        prompt_config: Optional[PromptConfig] = None,
        context_config: Optional[ContextConfig] = None,
        output_field: str = "qanswer",
    ):
        self.prompt = prompt
        self.backend = backend
        self.context_aggregation = context_aggregation
        self.prompt_config = prompt_config
        self.context_config = context_config
        self.output_field = output_field
        self.__post_init__()

    def __post_init__(self):
        if self.context_config is not None:
            if self.context_aggregation is not None:
                self.context_aggregation = ContextAggregationTransformer(
                    config=self.context_config, aggregate_func=self.context_aggregation
                )
            else:
                self.context_aggregation = ContextAggregationTransformer(
                    config=self.context_config
                )
        if isinstance(self.prompt, str):
            self.prompt = PromptTransformer(
                instruction=self.prompt,
                model_name_or_path=self.LLM._model_name_or_path,
                config=self.prompt_config,
                context_aggregation=self.context_aggregation,
                context_config=self.context_config,
            )

        self.prompt.set_output_attribute(self.backend._api_type)
        if self.prompt.expect_logits and not self.backend._support_logits:
            raise ValueError("The LLM does not support logits")
        elif self.prompt.expect_logits and self.LLM._support_logits:
            self.backend = self.backend.logit_generator()
        else:
            self.backend = self.backend.text_generator()

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        prompts = self.prompt(inp)
        outputs = self.backend(prompts)
        answers = self.prompt.answer_extraction(outputs)

        prompts[self.output_field] = answers
        return prompts
