from typing import Optional, Union, Iterable, List, Any

import pyterrier as pt
import pyterrier_alpha as pta
from fastchat.model import get_conversation_template


class PromptTransformer(pt.Transformer):
    def __init__(
        self,
        instruction: Union[callable, str] = None,
        model_name_or_path: str = None,
        system_message: Optional[str] = None,
        text_loader: Optional[callable] = None,
        conversation_template: Optional[Any] = None,
        api_type: Optional[str] = None,
        output_field: str = "prompt",
        input_fields: List[str] = ["query", "context"],
        expects_logits: bool = False,
        answer_extraction: Optional[callable] = None,
    ):
        self.instruction = instruction
        self.model_name_or_path = model_name_or_path
        self.system_message = system_message
        self.text_loader = text_loader
        self.output_field = output_field
        self.input_fields = input_fields
        self.conversation_template = conversation_template
        self.api_type = api_type
        self.expect_logits = expects_logits
        self.answer_extraction = answer_extraction or self.answer_extraction

        self.__post_init__()

    def __post_init__(self):
        self.conversation_template = (
            get_conversation_template(self.model_name_or_path)
            or self.conversation_template
        )
        if self.system_message is not None:
            # TODO: Set flag for if model supports system message
            self.conversation_template.set_system_message(self.system_message)

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
        current_prompt.append_message("user", instruction)
        return self.to_output(current_prompt)

    @pta.transform.by_query(add_ranks=False)
    def transform_iter(self, inp: Iterable[dict]) -> Iterable[dict]:
        return self.transform_by_query(inp)

    def transform_by_query(self, inp: Iterable[dict]) -> Iterable[dict]:
        inp = list(inp)
        qid = inp[0].get("qid", None)
        query = inp[0].get("query", None)
        if query is None and self.text_loader is not None:
            query = self.text_loader(qid)
            for i in inp:
                i["query"] = query
        fields = {k: v for k, v in inp[0].items() if k in self.input_fields}
        if (
            "text" in self.input_fields
            and "text" not in fields
            and self.text_loader is not None
        ):
            for i in inp:
                fields["text"] = self.text_loader(i["docno"])
        prompt = self.create_prompt(fields)

        return [{self.output_field: prompt, "qid": qid, "query_0": query}]


__all__ = ["PromptTransformer"]
