import torch
import pyterrier as pt
import pandas as pd

from ._base import AgenticRAG
from ... import VLLMBackend, HuggingFaceBackend


class SearchR1(AgenticRAG):

    DEFAULT_MODEL = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo"

    def __init__(
        self,
        retriever: pt.Transformer,
        backend: VLLMBackend | HuggingFaceBackend,
        top_k: int = 3,
        max_turn: int = 10,
        **kwargs,
    ) -> None:

        super().__init__(
            retriever,
            backend,
            prompt_template=self._get_prompt_template(backend),
            top_k=top_k,
            max_turn=max_turn,
            start_search_tag="<search>",
            end_search_tag="</search>",
            start_results_tag="<information>",
            end_results_tag="</information>",
            start_answer_tag="<answer>",
            end_answer_tag="</answer>",
            **kwargs,
        )

    @staticmethod
    def from_vllm(
        retriever: pt.Transformer,
        model: str = DEFAULT_MODEL,
        backend_args: dict | None = None,
        **kwargs,
    ) -> "SearchR1":
        if not backend_args:
            backend_args = {
                "model_args": {
                    "gpu_memory_utilization": 0.8,
                    "dtype": "bfloat16",
                    "max_model_len": 10240,
                },
                "generation_args": {
                    "temperature": 0.7,
                    "max_tokens": 1024,
                },
            }

        backend = VLLMBackend(model, **backend_args)

        return SearchR1(retriever, backend, **kwargs)

    @staticmethod
    def from_hf(
        retriever: pt.Transformer,
        model: str = DEFAULT_MODEL,
        backend_args: dict | None = None,
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

    def wrap_search_results(self, docs: pd.DataFrame) -> str:
        if not isinstance(docs, pd.DataFrame) or docs.empty:
            return f"{self.start_results_tag}{self.end_results_tag}"

        def _format_doc(idx, doc) -> str:
            title = doc.title.strip('"')
            text = doc.text
            return f'Doc {idx}(Title: "{title}") {text.removeprefix(title).lstrip()}'

        docs_str = "\n".join(_format_doc(idx, doc) for idx, doc in enumerate(docs.itertuples(), start=1))
        return f"\n\n{self.start_results_tag}{docs_str}{self.end_results_tag}\n\n"

    def extract_search_query(self, output: str) -> str | None:
        if not output:
            return None

        *head, query = output.rsplit(self.start_search_tag, maxsplit=1)
        if not head:
            return None

        query = query.split(self.end_search_tag, maxsplit=1)[0]

        return query.strip()

    def get_prompt(self, question: str) -> str:
        # Search-R1 was trained with this preprocessing
        question = question.strip()
        if question[-1] != '?':
            question += '?'

        return self.prompt_template.format(question=question)

    def _get_prompt_template(self, backend: HuggingFaceBackend | VLLMBackend) -> str:
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
