from typing import List, Union, Iterable

import pyterrier as pt
import pyterrier_alpha as pta
import pandas as pd
from fastchat import get_conversation_template


class PromptTransformer(pt.Transformer):
    def __init__(self,
                 system_message: str = None,
                 instruction: Union[callable, str] = None,
                 model_name_or_path: str = None,
                 conversation_template: str = None,
                 output_field: str = 'query',
                 relevant_fields: List[str] = ['query', 'context'],
                 api_type: str = None):
        assert model_name_or_path or conversation_template, "Either model_name_or_path or conversation_template must be provided"

        self.system_message = system_message
        self.instruction = instruction if isinstance(instruction, callable) else instruction.format
        self.conversation_template = get_conversation_template(model_name_or_path) or conversation_template
        if system_message:
            self.conversation_template.set_system_message(self.system_message)
        self.output_field = output_field
        self.relevant_fields = relevant_fields
        self.api_type = api_type
        self.output_attribute = {
            'openai': 'to_openai_api_messages',
            'gemini': 'to_gemini_api_messages',
            'vertex': 'to_vertex_api_messages',
            'reka': 'to_reka_api_messages',
        }[api_type] if api_type else 'get_prompt'

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

    def transform(self, inp: pd.DataFrame):
        inp[self.output_field] = inp.apply(lambda x: x[self.relevant_fields].to_dict(), axis=1)
        return inp


__all__ = ['PromptTransformer']
