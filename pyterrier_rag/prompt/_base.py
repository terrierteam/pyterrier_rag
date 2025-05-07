from typing import Optional, Union, Iterable, List, Any

import pyterrier as pt
import pyterrier_alpha as pta
from pyterrier_rag.prompt.wrapper import prompt
from fastchat.model import get_conversation_template
from fastchat.conversation import get_conv_template


class PromptTransformer(pt.Transformer):
    """
    Transformer that constructs and formats prompts for conversational LLMs.

    Parameters:
        instruction (callable|str): Template or function returning the instruction segment.
        model_name_or_path (str, optional): Model identifier for selecting conversation template.
        system_message (str, optional): System context message for the conversation.
        conversation_template (Any, optional): Preconfigured conversation template.
        api_type (str, optional): API format: 'openai','gemini','vertex','reka'.
        output_field (str): Field name to store the generated prompt.
        input_fields (List[str]): Input record fields required to build the prompt.
        expects_logits (bool): Indicator for logit-based backends.
        answer_extraction (callable, optional): Function to parse model outputs.
        raw_instruction (bool): If True, returns raw instruction without template.
    """
    def __init__(
        self,
        instruction: Union[callable, str] = None,
        model_name_or_path: str = None,
        system_message: Optional[str] = None,
        conversation_template: Optional[Any] = None,
        api_type: Optional[str] = None,
        output_field: str = "prompt",
        input_fields: List[str] = ["query", "qcontext"],
        expects_logits: bool = False,
        answer_extraction: Optional[callable] = None,
        raw_instruction: bool = False,
    ):
        self.instruction = instruction
        self.model_name_or_path = model_name_or_path
        self.system_message = system_message
        self.output_field = output_field
        self.input_fields = input_fields
        self.conversation_template = conversation_template
        self.api_type = api_type
        self.expect_logits = expects_logits
        self.answer_extraction = answer_extraction or self.answer_extraction
        self.raw_instruction = raw_instruction

        self.__post_init__()

    def __post_init__(self):
        if type(self.instruction) is str:
            self.instruction = prompt(self.instruction)
        if self.model_name_or_path is not None:
            self.conversation_template = (
                get_conversation_template(self.model_name_or_path) or self.conversation_template
            )
        if self.conversation_template is None:
            self.conversation_template = get_conv_template("raw")
            self.raw_instruction = True
        if self.system_message is not None:
            # TODO: Set flag for if model supports system message
            self.conversation_template.set_system_message(self.system_message)

        roles = self.conversation_template.roles
        if len(roles) < 2:
            self.user_role, self.llm_role = "user", "assistant"
        else:
            self.user_role = roles[0]
            self.llm_role = roles[1]

        self.output_attribute = (
            {
                "openai": "to_openai_api_messages",
                "gemini": "to_gemini_api_messages",
                "vertex": "to_vertex_api_messages",
                "reka": "to_reka_api_messages",
            }[self.api_type]
            if self.api_type
            else "get_prompt"
        )

    def answer_extraction(self, output):
        return output

    def set_output_attribute(self, api_type: str = None):
        self.output_attribute = (
            {
                "openai": "to_openai_api_messages",
                "gemini": "to_gemini_api_messages",
                "vertex": "to_vertex_api_messages",
                "reka": "to_reka_api_messages",
            }[api_type]
            if api_type
            else "get_prompt"
        )

    @property
    def prompt(self):
        return self.conversation_template.copy()

    def to_output(self, prompt) -> Union[str, List[dict]]:
        return getattr(prompt, self.output_attribute)()

    def create_prompt(self, fields: dict) -> Union[str, List[dict]]:
        current_prompt = self.prompt
        instruction = self.instruction(**fields)
        if self.raw_instruction:
            return instruction
        current_prompt.append_message(self.user_role, instruction)
        return self.to_output(current_prompt)

    @pta.transform.by_query(add_ranks=False)
    def transform_iter(self, inp: Iterable[dict]) -> Iterable[dict]:
        return self.transform_by_query(inp)

    def transform_by_query(self, inp: Iterable[dict]) -> Iterable[dict]:
        inp = list(inp)
        qid = inp[0].get("qid", None)
        query = inp[0].get("query", None)
        fields = {k: v for k, v in inp[0].items() if k in self.input_fields}
        if any([f not in fields for f in self.input_fields]):
            message = f"Expected {self.input_fields} but recieved {fields}"
            raise ValueError(message)
        prompt = self.create_prompt(fields)
        return [{self.output_field: prompt, "qid": qid, "query_0": query}]


__all__ = ["PromptTransformer"]
