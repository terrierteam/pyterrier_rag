from abc import ABC, abstractmethod
from typing import Iterable, Tuple, Union

import pyterrier as pt
import pyterrier_alpha as pta
import torch
from transformers import GenerationConfig

GENERIC_PROMPT = "Use the context information to answer the Question: \n Context: {context} \n Question: {query} \n Answer:"


class Reader(pt.Transformer, ABC):

    def __init__(
        self,
        *,
        batch_size: int = 4,
        text_field: str = 'text',
        text_max_length: int = 512,
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
        self.text_max_length = text_max_length
        self.num_context = num_context
        self.max_new_tokens = max_new_tokens
        self.verbose = verbose
        self.kwargs = kwargs

        if generation_config is None:
            # use greedy decoding by default
            self.generation_config = GenerationConfig(
                max_new_tokens = self.max_new_tokens,
                temperature=1.0,
                do_sample = False,
                num_beams = 1
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

    @abstractmethod
    def transform_by_query(self, inp: Iterable[dict]) -> Iterable[dict]:
        pass

    def get_context_by_query(self, inp: Iterable[dict]) -> Iterable[Union[str, Tuple[str]]]:
        """Return at most self.num_context retrieved context.
        """
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
