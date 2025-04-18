# Modified from https://github.com/sunnynexus/Search-o1
# Changes made by Jinyuan on 2025-04-14

import pyterrier as pt
from pyterrier_rag.backend import CausalLMLLM, LLM


class SearchO1(pt.Transformer):
    def __init__(
        self,
        retriever: pt.Transformer,
        LLM: LLM,
        input_field: str = "query",
        output_field: str = "qanswer",
        mode: str = "gen",
        max_search_limit: int = 5,
        max_iterations: int = -1,
    ):
        if not isinstance(LLM, CausalLMLLM):
            raise ValueError(
                "The LLM for Search-O1 must currently be a CausalLMLLM instance."
            )
        self.retriever = retriever
        self.LLM = LLM
        self.input_field = input_field
        self.output_field = output_field
        self.mode = mode
        self.max_search_limit = max_search_limit
        self.max_iterations = max_iterations

        self.__post_init__()

    def __post_init__(self):
        self.LLM.generation_config["stopping_criteria"] = [
            StopWordCriteria(
                self.tokenizer,
                prompt_length,
                [END_SEARCH_QUERY, self.tokenizer.eos_token],
            )
        ]

    def _make_default_context_config(self):
        return ContextConfig(
            in_fields=["text"],
            out_field="context",
            tokenizer=self.reader.tokenizer,
            max_length=self.reader.max_input_length,
            max_elements=self.max_docs,
            intermediate_format=text_format,
        )
