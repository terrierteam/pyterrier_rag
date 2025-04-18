from typing import Optional, Union, Iterable, List

import pyterrier as pt
import pyterrier_alpha as pta
from fastchat.model import get_conversation_template

from pyterrier_rag.prompt._config import PromptConfig, ContextConfig
from pyterrier_rag.prompt._context_aggregation import ContextAggregationTransformer


class PromptTransformer(pt.Transformer):
    def __init__(
        self,
        instruction: Union[callable, str] = None,
        model_name_or_path: str = None,
        system_message: Optional[str] = None,
        text_loader: Optional[callable] = None,
        output_field: str = "prompt",
        input_fields: List[str] = ["query", "context"],
        config: Optional[PromptConfig] = None,
        context_aggregation: Optional[callable] = None,
        context_config: Optional[ContextConfig] = None,
        expects_logits: bool = False,
        answer_extraction: Optional[callable] = None,
    ):
        self.text_loader = text_loader
        if config is None:
            config = PromptConfig(
                instruction=instruction,
                model_name_or_path=model_name_or_path,
                system_message=system_message,
                output_field=output_field,
                input_fields=input_fields,
            )
        if context_aggregation is None and context_config is None:
            self.use_context = False
        elif context_config is None and context_aggregation is not None:
            context_config = ContextConfig(aggregate_func=context_aggregation)
            self.use_context = True
        elif context_config is not None:
            self.use_context = True
        self.config = config
        self.context_config = context_config
        self.output_field = config.output_field
        self.config.input_fields = input_fields
        self.api_type = config.api_type
        self.expect_logits = expects_logits
        self.answer_extraction = answer_extraction or self.answer_extraction

        self.__post_init__()

    def __post_init__(self):
        self.conversation_template = (
            get_conversation_template(self.config.model_name_or_path)
            or self.config.conversation_template
        )
        if self.config.system_message is not None:
            # TODO: Set flag for if model supports system message
            self.conversation_template.set_system_message(self.config.system_message)
        self.instruction = self.config.instruction

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

        self.context_aggregation = (
            ContextAggregationTransformer(config=self.context_config)
            if self.use_context
            else None
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
        fields = {k: v for k, v in inp[0].items() if k in self.relevant_fields}
        if (
            "text" in self.relevant_fields
            and "text" not in fields
            and self.text_loader is not None
        ):
            for i in inp:
                fields["text"] = self.text_loader(i["docno"])
        if self.use_context:
            context = self.context_aggregation.transform_by_query(inp)["context"]
            fields["context"] = context
        prompt = self.create_prompt(fields)

        return [{self.output_field: prompt, "qid": qid, "query_0": query}]


__all__ = ["PromptTransformer"]
