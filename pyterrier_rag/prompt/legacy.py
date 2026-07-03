import warnings
from typing import Any, Callable, List, Optional, Union

import pandas as pd
import pyterrier as pt
import pyterrier_alpha as pta

from pyterrier_rag._util import concat
from pyterrier_rag.prompt.jinja import jinja_formatter


def _warn_legacy(symbol: str, replacement: str) -> None:
    warnings.warn(
        (
            f"`pyterrier_rag.prompt.{symbol}` is deprecated and will be removed in a future release. "
            f"Use {replacement} instead."
        ),
        DeprecationWarning,
        stacklevel=3,
    )


def prompt(template: str):
    _warn_legacy("prompt", "`pyterrier_rag.prompt.jinja_formatter`")
    return jinja_formatter(template)


def score_sort(inp: List[dict]):
    if "score" in inp[0]:
        return sorted(inp, key=lambda x: x["score"], reverse=True)
    return inp


class Concatenator(pt.Transformer):
    """
    Legacy compatibility shim for the removed context aggregation pipeline component.
    """

    def __init__(
        self,
        in_fields: Optional[List[str]] = ["text"],
        out_field: Optional[str] = "qcontext",
        text_loader: Optional[Callable] = None,
        intermediate_format: Optional[Callable] = None,
        tokenizer: Optional[Any] = None,
        max_length: Optional[int] = -1,
        max_elements: Optional[int] = -1,
        max_per_context: Optional[int] = -1,
        truncation_rate: Optional[int] = 50,
        aggregate_func: Optional[Callable] = None,
        ordering_func: Optional[Callable] = score_sort,
    ):
        super().__init__()
        _warn_legacy(
            "Concatenator",
            (
                "`Reader` prompt templates that consume grouped docs directly "
                "(drop the explicit `>> Concatenator()` pipeline stage)"
            ),
        )
        self.in_fields = in_fields
        self.out_field = out_field
        self.aggregate_func = aggregate_func
        self.text_loader = text_loader
        self.intermediate_format = intermediate_format
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_elements = max_elements
        self.max_per_context = max_per_context
        self.truncation_rate = truncation_rate
        self.ordering_func = ordering_func

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        pta.validate.columns(inp, includes=["qid"] + self.in_fields)
        output_frame = pta.DataFrameBuilder([self.out_field, "qid", "query"])

        if inp is None or inp.empty:
            return output_frame.to_df()

        for qid, group in inp.groupby("qid"):
            inp_records = group.to_dict(orient="records")
            qid = inp_records[0].get("qid", None)
            query = inp_records[0].get("query", None)
            if self.ordering_func is not None:
                inp_records = self.ordering_func(inp_records)
            relevant = [{k: v for k, v in rec.items() if k in self.in_fields} for rec in inp_records]
            if "text" in self.in_fields and "text" not in inp_records[0].keys():
                if self.text_loader is None:
                    raise ValueError("Cannot retrieve text without a text loader")
                for doc, row in zip(relevant, inp_records):
                    doc["text"] = self.text_loader(row["docno"])

            if self.aggregate_func is not None:
                context = self.aggregate_func(relevant)
            else:
                context = concat(
                    relevant,
                    intermediate_format=self.intermediate_format,
                    tokenizer=self.tokenizer,
                    max_length=self.max_length,
                    max_elements=self.max_elements,
                    max_per_context=self.max_per_context,
                    truncation_rate=self.truncation_rate,
                )
            output_frame.extend({self.out_field: context, "qid": qid, "query": query})
        return output_frame.to_df()


class PromptTransformer(pt.Transformer):
    """
    Legacy compatibility shim for the removed PromptTransformer.
    """

    def __init__(
        self,
        instruction: Union[Callable, str] = None,
        model_name_or_path: str = None,
        system_message: Optional[str] = None,
        conversation_template: Optional[Any] = None,
        api_type: Optional[str] = None,
        output_field: str = "prompt",
        input_fields: List[str] = ["query", "qcontext"],
        expects_logprobs: bool = False,
        answer_extraction: Optional[Callable] = None,
        raw_instruction: bool = False,
    ):
        _warn_legacy(
            "PromptTransformer",
            (
                "`Reader(prompt=...)` with `jinja_formatter(...)` templates "
                "or prompt callables"
            ),
        )
        self.instruction = jinja_formatter(instruction) if isinstance(instruction, str) else instruction
        self.model_name_or_path = model_name_or_path
        self.system_message = system_message
        self.conversation_template = conversation_template
        self.api_type = api_type
        self.output_field = output_field
        self.input_fields = input_fields
        self.expects_logprobs = expects_logprobs
        self.answer_extraction = answer_extraction or self._default_answer_extraction
        self.raw_instruction = raw_instruction
        self._supports_message_input = False

        if self.expects_logprobs:
            warnings.warn(
                "`PromptTransformer(expects_logprobs=True)` is deprecated; use backend-level logprob generation.",
                DeprecationWarning,
                stacklevel=2,
            )

    def _default_answer_extraction(self, output):
        return output

    def set_output_attribute(self, supports_message_input: bool):
        self._supports_message_input = supports_message_input

    def create_prompt(self, fields: dict) -> Union[str, List[dict]]:
        instruction = self.instruction(**fields)
        if self.raw_instruction:
            return instruction
        if self._supports_message_input:
            messages = []
            if self.system_message is not None:
                messages.append({"role": "system", "content": self.system_message})
            messages.append({"role": "user", "content": instruction})
            return messages
        if self.system_message is not None:
            return self.system_message + "\n\n" + instruction
        return instruction

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        pta.validate.columns(inp, includes=["qid"] + self.input_fields)
        output_frame = pta.DataFrameBuilder([self.output_field, "qid", "query_0"])

        if inp is None or inp.empty:
            return output_frame.to_df()

        for qid, group in inp.groupby("qid"):
            records = group.to_dict(orient="records")
            query = records[0].get("query", None)
            fields = {k: v for k, v in records[0].items() if k in self.input_fields}
            if any(field not in fields for field in self.input_fields):
                raise ValueError(f"Expected {self.input_fields} but received {fields}")
            prompt_text = self.create_prompt(fields)
            output_frame.extend({self.output_field: prompt_text, "qid": qid, "query_0": query})

        return output_frame.to_df()


__all__ = ["prompt", "score_sort", "Concatenator", "PromptTransformer"]
