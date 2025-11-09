from typing import Literal

import torch
import pyterrier as pt
import pandas as pd

from ._base import AgenticRAG
from ... import VLLMBackend, HuggingFaceBackend


class R1Searcher(AgenticRAG):
    """R1-Searcher.

    Args:
        prompt_type (Literal["v1", "v2", "v3"], optional): prompt template.
            - v0: Single-step reasoning. When encountering uncertain knowledge, use <|begin_of_query|>keyword<|end_of_query|> to search. Suitable for general Q&A.
            - v1: Multi-step/sub-question reasoning. The question is split into sub-questions, each of which is searched using <|begin_of_query|>kw1\tkw2<|end_of_query|>. Keywords are separated by tabs. Suitable for multi-hop/complex Q&A.
            - v2: Multi-step reasoning + keyword search. The search query only allows keyword lists (separated by \t), not complete sentences. Suitable for keyword-only search scenarios.
            - v3: Judgment reasoning. Designed for yes/no questions. The answer after reasoning must be yes or no. The search method is the same as v0. Suitable for judgment-type Q&A.
            Defaults to 'v1'.
    """

    DEFAULT_MODEL = "XXsongLALA/Qwen-2.5-7B-base-RAG-RL"

    def __init__(
        self,
        retriever: pt.Transformer,
        backend: VLLMBackend | HuggingFaceBackend,
        top_k=5,
        max_turn=10,
        prompt_type: Literal["v1", "v2", "v3"] = "v1",
        **kwargs,
    ) -> None:
        super().__init__(
            retriever,
            backend,
            top_k=top_k,
            max_turn=max_turn,
            prompt_template=self._get_prompt_template(prompt_type),
            start_search_tag="<|begin_of_query|>",
            end_search_tag="<|end_of_query|>",
            start_results_tag="<|begin_of_documents|>",
            end_results_tag="<|end_of_documents|>",
            start_answer_tag="<answer>",
            end_answer_tag="</answer>",
            **kwargs,
        )

        self.stop_sequences = ["<|im_end|>", "<|endoftext|>", "<|end_of_query|>", "</answer>"]

    @staticmethod
    def from_vllm(
        retriever: pt.Transformer,
        model: str = DEFAULT_MODEL,
        backend_args: dict | None = None,
        **kwargs,
    ):
        if not backend_args:
            backend_args = {
                "model_args": {
                    "dtype": "bfloat16",
                    "gpu_memory_utilization": 0.8,
                    "max_model_len": 16384,
                    "trust_remote_code": True,
                },
                "generation_args": {
                    "temperature": 0,
                    "top_p": 0.95,
                    "max_tokens": 512,
                },
            }

        backend = VLLMBackend(model, **backend_args)
        return R1Searcher(retriever, backend=backend, **kwargs)

    @staticmethod
    def from_hf(
        retriever: pt.Transformer,
        model: str = DEFAULT_MODEL,
        backend_args: dict | None = None,
        **kwargs,
    ):
        if not backend_args:
            backend_args = {
                "model_args": {
                    "torch_dtype": torch.bfloat16,
                    "device_map": "auto",
                    "trust_remote_code": True,
                },
                "generation_args": {
                    "max_new_tokens": 512,
                    "do_sample": False,
                },
            }

        backend = HuggingFaceBackend(model, **backend_args)
        backend.tokenizer.padding_side = "left"
        # to suppress warning
        backend._model.generation_config.pad_token_id = backend.tokenizer.pad_token_id
        return R1Searcher(retriever, backend=backend, **kwargs)

    def wrap_search_results(self, docs: pd.DataFrame) -> str:
        if not isinstance(docs, pd.DataFrame) or docs.empty:
            return f"\n\n{self.start_results_tag}\nNone{self.end_results_tag}\n\n"

        docs_str = "".join(f"({idx}){doc.text}\n" for idx, doc in enumerate(docs.itertuples(), start=1))
        return f"\n\n{self.start_results_tag}\n{docs_str}{self.end_results_tag}\n\n"

    def _get_prompt_template(self, prompt_type: str):
        if prompt_type == "v0":
            return V0_PROMPT
        elif prompt_type == "v1":
            return V1_PROMPT
        elif prompt_type == "v2":
            return V2_PROMPT
        elif prompt_type == "v3":
            return V3_PROMPT
        else:
            raise ValueError(f"unkown {prompt_type=}")


V0_PROMPT = """The User asks a question, and the Assistant solves it.
The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer.
The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
During the thinking process, the Assistant can perform searching for uncertain knowledge if necessary with the format of "<|begin_of_query|> search query (only keywords) here <|end_of_query|>". **A query must involve only a single triple**.
Then, the system will provide the Assistant with helpful information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>".\n\nUser:{question}\nAssistant: <think>"""

V1_PROMPT = """The User asks a question, and the Assistant solves it.
The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer.
The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
During the reasoning process, the Assistant will break down the original question into sub-questions and address them step by step.
For each sub-question, **the Assistant can perform searching** for uncertain knowledge using the format: "<|begin_of_query|> keyword1\tkeyword2\t... <|end_of_query|>".
**The query must consist of straightforward and essential keywords separated by "\t"**. Furthermore, **the query must involve only a single triple to address a sub-question**.
Then, the search system will provide the Assistant with relevant information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>".

User:{question}
Assistant: <think>"""

V2_PROMPT = """The User asks a question, and the Assistant solves it.
The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer.
The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
During the thinking process, **the Assistant can perform searching** for uncertain knowledge if necessary with the format of "<|begin_of_query|> search query (only list keywords separated by "\t" instead of the complete sentence , such as **"keyword_1 \t keyword_2 \t..."**)<|end_of_query|>". **A query must involve only a single triple**.
Then, the search system will provide the Assistant with the retrieval information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>".

User:{question}
Assistant: <think>"""

V3_PROMPT = """The User asks a **Judgment question**, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer. The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here (yes or no) </answer>". During the thinking process, the Assistant can perform searching for uncertain knowledge if necessary with the format of "<|begin_of_query|> search query (only keywords) here <|end_of_query|>". Then, the system will provide the Assistant with helpful information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>". The final answer **must be yes or no**.\n\nUser:{question}\nAssistant: <think>"""
