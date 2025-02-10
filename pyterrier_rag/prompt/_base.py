from typing import Optional, Union, Iterable

import pyterrier as pt
import pyterrier_alpha as pta
from fastchat import get_conversation_template

from ._config import PromptConfig, ContextConfig
from ._context_aggregation import ContextAggregationTransformer


class PromptTransformer(pt.Transformer):
    def __init__(self,
                 instruction: Union[callable, str] = None,
                 model_name_or_path: str = None,
                 system_message: Optional[str] = None,
                 config: Optional[PromptConfig] = None,
                 context_aggregation: Optional[callable] = None,
                 context_config: Optional[ContextConfig] = None,):

        if config is None:
            config = PromptConfig(
                instruction=instruction,
                model_name_or_path=model_name_or_path,
                system_message=system_message
            )
        if context_config is None:
            context_config = ContextConfig(aggregate_func=context_aggregation)
        self.config = config
        self.context_config = context_config
        self.output_field = config.output_field
        self.relevant_fields = config.input_fields
        self.api_type = config.api_type

    def __post_init__(self):
        self.conversation_template = get_conversation_template(self.config.model_name_or_path) or self.config.conversation_template
        if self.config.system_message is not None:
            self.conversation_template.set_system_message(self.config.system_message)

        self.output_attribute = {
            'openai': 'to_openai_api_messages',
            'gemini': 'to_gemini_api_messages',
            'vertex': 'to_vertex_api_messages',
            'reka': 'to_reka_api_messages',
        }[self.api_type] if self.api_type else 'get_prompt'

        self.context_aggregation = ContextAggregationTransformer(**self.context_config.__dict__)

    @property
    def prompt(self):
        return self.conversation_template.copy()

    def to_output(self, prompt):
        return getattr(prompt, self.output_attribute)()

    def create_prompt(self, fields: dict):
        current_prompt = self.prompt
        instruction = self.instruction(**fields)
        current_prompt.append_message('user', instruction)
        return current_prompt

    @pta.transform.by_query(add_ranks=False)
    def transform_iter(self, inp: Iterable[dict]) -> Iterable[dict]:
        return self.transform_by_query(inp)

    def transform_by_query(self, inp: Iterable[dict]) -> Iterable[dict]:
        all_fields = list(inp)
        fields = {field: all_fields[0][field] for field in self.relevant_fields}
        prompt = self.create_prompt(fields)

        return {self.output_field: prompt, **inp[0]}


__all__ = ['PromptTransformer']
