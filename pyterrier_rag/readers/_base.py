from abc import ABC, abstractmethod
from typing import Iterable, Tuple, Union

import pyterrier as pt
import pyterrier_alpha as pta
import torch
import pandas as pd
from transformers import GenerationConfig

GENERIC_PROMPT = "Use the context information to answer the Question: \n Context: {context} \n Question: {query} \n Answer:"


class Reader(pt.Transformer, ABC):

    def __init__(
        self,
        *,
        batch_size: int = 4,
        text_field: str = "text",
        max_input_length: int = 512,
        num_context: Union[int, str] = "auto",
        max_new_tokens: int = 32,
        generation_config: GenerationConfig = None,
        verbose: bool = False,
        device: Union[str, torch.device] = None,
        **kwargs
    ):
        super().__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.text_field = text_field
        self.max_input_length = max_input_length
        self.num_context = num_context
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.verbose = verbose
        self.kwargs = kwargs

        if generation_config is None:
            # use greedy decoding by default
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_new_tokens,
                temperature=1.0,
                do_sample=False,
                num_beams=1,
            )
        else:
            self.generation_config = generation_config

    @property
    def is_openai(self):
        return False

    # TODO: couldn't pass self.verbose to pta.transform.by_query
    @pta.transform.by_query(add_ranks=False)
    def transform_iter(self, inp: Iterable[dict]) -> Iterable[dict]:
        return self.transform_by_query(inp)

    def transform_by_query(self, inp: Iterable[dict]) -> Iterable[dict]:
        inp = list(inp)
        qid = inp[0]["qid"]
        query = inp[0]["query"]
        outputs = self._generate(query)

        return [{"qid": qid, "query": query, "qanswer": outputs}]

    def transform(self, inp : pd.DataFrame) -> pd.DataFrame:
        inp = inp.drop_duplicates(subset='qid')
        qids = inp['qid'].tolist()
        queries = inp['query'].tolist()
        qanswers = self.generate(queries)

        return pd.DataFrame({'qid': qids, 'query': queries, 'qanswer': qanswers})

    def get_context_by_query(
        self, inp: Iterable[dict]
    ) -> Iterable[Union[str, Tuple[str]]]:
        """Return at most self.num_context retrieved context."""
        if self.num_context and inp:
            num = len(inp) if self.num_context == "auto" else self.num_context
            if "score" in inp[0]:
                inp = sorted(inp, key=lambda x: x["score"], reverse=True)
            if "title" in inp[0]:
                context = [(item["title"], item[self.text_field]) for item in inp]
            else:
                context = [item[self.text_field] for item in inp]
            context = context[:num]
        else:
            context = None
        return context
