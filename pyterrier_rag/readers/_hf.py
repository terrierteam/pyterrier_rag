import torch 
from typing import Any, List

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, StoppingCriteria 

from . import _content_aggregation as content_aggregation
from ._base import GENERIC_PROMPT, Reader


class HuggingFaceReader(Reader):
    _prompt = GENERIC_PROMPT
    _model_class = None
    def __init__(self,
                    model_name_or_path: str,
                    model_args: dict = {},
                    generation_args: dict = None,
                    context_aggregation: str = 'concat',
                    prompt: Any = None,
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
            self._model_name_or_path = model_name_or_path
            self._model = None if self._model_class is None else self._model_class.from_pretrained(model_name_or_path, **model_args).to(self.device).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

            if context_aggregation not in content_aggregation.__all__:
                raise ValueError(f"context_aggregation must be one of {content_aggregation.__all__}")
            self._context_aggregation = getattr(content_aggregation, context_aggregation)
            self._prompt = prompt or self._prompt

            if isinstance(self._prompt, str):
                self._prompt = self._prompt.format

            if generation_args is None:
                generation_args = {
                    'max_new_tokens': self.max_new_tokens,
                    'temperature': 1.0,
                    'do_sample': False,
                    'num_beams': 1,
                }
            self._generation_args = generation_args
            self.model = self._model

    def generate(self, inps : List[str]) -> List[str]:
        assert self.model is not None, "Model is not loaded, you should instantiate a subclass of HFModel"
        inputs = self.tokenizer(inps, return_tensors='pt', padding=True, truncation=True, max_length=2048) # TODO - please fix AP: self.max_input_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model.generate(**inputs, **self._generation_args)

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


    def transform_by_query(self, inp: List[dict]) -> List[dict]:
        inp = list(inp)
        qid = inp[0]["qid"]
        query = inp[0]["query"]

        context = self.get_context_by_query(inp)
        aggregate_context = self._context_aggregation(context)
        input_texts = self._prompt(query=query, context=aggregate_context)
        outputs = self.generate([input_texts])
        return [{"qid": qid, "query": query, "qanswer": outputs[0]}]


class CausalLMReader(HuggingFaceReader):
    _model_class = AutoModelForCausalLM


class Seq2SeqLMReader(HuggingFaceReader):
    _model_class = AutoModelForSeq2SeqLM


class StopWordCriteria(StoppingCriteria):

    def __init__(self, tokenizer: AutoTokenizer, prompt_size: int, stop_words: List[str] = [], check_every: int = 1):
        """
        Initializes the StopWordCriteria with the necessary parameters for checking stop words during text generation.
        
        Parameters:
            tokenizer (AutoTokenizer): The tokenizer for encoding prompts and stop words.
            # prompts (List[str]): Initial prompts used for generation, needed to determine where generated text begins.
            prompt_size (int): used to determine where the generated text begins. (目前只支持left padding)
            stop_words (List[str]): Words that trigger the stopping of generation when detected.
            check_every (int): Frequency of checking for stop words in the token stream (a performance optimization, use 1 to cut it out).
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_size = prompt_size

        self.stop_words = stop_words
        self.max_stop_word_size = max((self.tokenizer.encode(word, return_tensors="pt").size(-1) for word in stop_words), default=0)
        self.check_every = check_every

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """
        Determines whether to stop generation based on the presence of stop words.
        
        Stops if a stop word is found in *all* batch elements *and* the sequence length is a multiple of `check_every`.
        Note: Delay in stopping may occur if `check_every > 1`.

        Parameters:
            input_ids (torch.LongTensor): Generated token IDs.
            scores (torch.FloatTensor): Generation scores for each token. Not used here.
        
        Returns:
            bool: True to stop generation, False to continue.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Skip check if no stop words are defined or it is not yet time to check
        results = torch.zeros((input_ids.shape[0],), dtype=torch.bool).to(device)

        if (len(self.stop_words) == 0) or (seq_len % self.check_every != 0):
            return results
        
        for i in range(batch_size):
            # Calculate starting index for new tokens
            prompt_size = self.prompt_size
            max_new_tokens = (2 * self.max_stop_word_size) + self.check_every
            latest_tokens = input_ids[i, prompt_size:][-max_new_tokens:]
            if any([word in self.tokenizer.decode(latest_tokens, skip_special_tokens=True) for word in self.stop_words]):
                results[i] = True
            
        return results

