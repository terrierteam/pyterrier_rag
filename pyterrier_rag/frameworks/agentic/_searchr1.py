from typing import Optional

import torch
import pyterrier as pt

from ._base import AgenticRAG
from ... import VLLMBackend, HuggingFaceBackend


class SearchR1(AgenticRAG):

    DEFAULT_MODEL = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo"

    @staticmethod
    def from_vllm(
        retriever: pt.Transformer,
        model: str = DEFAULT_MODEL,
        backend_args: Optional[dict] = None,
        **kwargs,
    ) -> "SearchR1":
        if not backend_args:
            backend_args = {
                "generation_args": {
                    "temperature": 0.7, 
                    "max_tokens": 1024,
                }
            }

        backend = VLLMBackend(model, **backend_args)

        return SearchR1(retriever, backend, **kwargs)

    @staticmethod
    def from_hf(
        retriever: pt.Transformer,
        model: str = DEFAULT_MODEL,
        backend_args: Optional[dict] = None,
        **kwargs,
    ) -> "SearchR1":
        if not backend_args:
            # as in official example
            backend_args = {
                "model_args": {
                    "torch_dtype": torch.bfloat16,
                    "device_map": "auto",
                },
                "generation_args": {
                    "temperature": 0.7,  
                    "max_new_tokens": 1024,
                    "do_sample": True,
                },
            }

        backend = HuggingFaceBackend(model, **backend_args)
        backend.tokenizer.padding_side = "left"
        # to suppress warning
        backend._model.generation_config.pad_token_id = backend.tokenizer.pad_token_id

        return SearchR1(retriever, backend, **kwargs)

    def __init__(self, retriever, backend, temperature=0.7, top_k=3, max_turn=10, max_tokens=1024, **kwargs):

        super().__init__(
            retriever,
            backend,
            prompt=self._get_prompt(backend),
            top_k=top_k,
            max_turn=max_turn,
            max_tokens=max_tokens,
            temperature=temperature,
            start_search_tag="<search>",
            end_search_tag="</search>",
            start_results_tag="<information>",
            end_results_tag="</information>",
            **kwargs,
        )

    def _get_prompt(self, backend: HuggingFaceBackend | VLLMBackend) -> str:
        if isinstance(backend, HuggingFaceBackend):
            tokenizer = backend.tokenizer
        elif isinstance(backend, VLLMBackend):
            tokenizer = backend.model.get_tokenizer()
        else:
            raise NotImplementedError(f"unsupported backend: {type(backend)}")

        prompt = """Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

        if tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )

        return prompt
