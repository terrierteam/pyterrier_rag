from typing import Any, Iterable
from ._base import Reader, GENERIC_PROMPT
from . import _content_aggregation as content_aggregation
from .._optional import is_vllm_availible


class VLLMReader(Reader):
    _prompt = GENERIC_PROMPT
    def __init__(self,
                 model_name_or_path: str,
                 model_args : dict = {},
                 generation_args : dict = None,
                 context_aggregation : str = 'concat',
                 prompt : Any = None,
                 batch_size: int = 4,
                 text_field: str = 'text',
                 text_max_length: int = 512,
                 num_context: int = 5,
                 max_new_tokens: int = 32,
                 verbose: bool = False,
                 **kwargs
                ):
        super().__init__(batch_size=batch_size,
                         text_field=text_field,
                         text_max_length=text_max_length,
                         num_context=num_context,
                         max_new_tokens=max_new_tokens,
                         generation_config=None,
                         verbose=verbose,
                         **kwargs)
        if not is_vllm_availible():
            raise ImportError("Please install vllm to use VLLMReader")
        from vllm import LLM, SamplingParams, EngineArgs, LLMEngine
        self._model_name_or_path = model_name_or_path
        self._args = EngineArgs(model=model_name_or_path, **model_args)
        self._model = LLMEngine.from_engine_args(self._args)

        if context_aggregation not in content_aggregation.__all__:
            raise ValueError(f"context_aggregation must be one of {content_aggregation.__all__}")
        self._context_aggregation = getattr(content_aggregation, context_aggregation)
        self._prompt = prompt or self._prompt

        if isinstance(self._prompt, str):
            self._prompt = self._prompt.format

        if generation_args is None:
            generation_args = {
                'max_new_tokens' : self.max_new_tokens,
                'temperature' : 1.0,
                'do_sample' : False,
                'num_beams' : 1,
            }
        self._generation_args = SamplingParams(**generation_args)
        self.model = LLM(self._model, self._generation_args)

    def transform_by_query(self, inp: Iterable[dict]) -> Iterable[dict]:
        inp = list(inp)
        qid = inp[0]["qid"]
        query = inp[0]["query"]
       
        context = self.get_context_by_query(inp)
        aggregate_context = self._context_aggregation(context)
        input_texts = self._prompt(query=query, context=aggregate_context)
        outputs = self.model.generate([input_texts], self._generation_args)

        return [{"qid": qid, "query": query, "qanswer": outputs[0][0].text}]
