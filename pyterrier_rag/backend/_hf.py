from typing import Optional, List, Union, Dict

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    StoppingCriteria,
)
import torch

from pyterrier_rag.backend._base import Backend, BackendOutput


class HuggingFaceBackend(Backend):
    """
        Backend implementation using a HuggingFace Transformer model.

        .. cite.dblp:: journals/corr/abs-1910-03771

        Parameters:
            model_id (str): Identifier or path of the pretrained model.
            model_args (dict): Arguments passed to `from_pretrained` for model instantiation.
            generation_args (dict): Parameters controlling text generation.
            max_input_length (int): Maximum token length for inputs (defaults to model config).
            max_new_tokens (int): Maximum number of tokens to generate per input.
            verbose (bool): Flag to enable verbose logging.
    """
    supports_logprobs = False # TODO: add support for logprobs
    _model_class = AutoModelForCausalLM
    _remove_prompt = True

    def __init__(
        self,
        model_id: str,
        *,
        model_args: dict = {},
        generation_args: dict = None,
        max_input_length: int = None,
        max_new_tokens: int = 32,
        logprobs_topk: int = 20,
        verbose: bool = False,
        device: Union[str, torch.device] = None,
    ):
        super().__init__(
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            verbose=verbose,
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self._model = (
            None
            if self._model_class is None
            else self._model_class.from_pretrained(model_id, **model_args).to(self.device).eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self._model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        max_position_embeddings = getattr(self._model.config, "max_position_embeddings", None)
        self.max_input_length = max_input_length or max_position_embeddings
        self.logprobs_topk = logprobs_topk

        if generation_args is None:
            generation_args = {
                "max_new_tokens": self.max_new_tokens,
                "temperature": 1.0,
                "do_sample": False,
                "num_beams": 1,
            }
        self._generation_args = generation_args

    @torch.no_grad()
    def generate(
        self,
        inps: Union[List[str], List[List[dict]]],
        *,
        return_logprobs: bool = False,
        max_new_tokens: Optional[int] = None,
        num_responses: int = 1,
    ) -> List[BackendOutput]:
        if not isinstance(inps[0], str):
            raise ValueError(f'{self!r} only supports str inputs to generate')
        if return_logprobs:
            raise ValueError(f'{self!r} does not support logprobs generation')
        if num_responses != 1:
            raise ValueError(f'{self!r} does not support num_responses > 1')
        # Tokenize inputs
        inputs = self.tokenizer(
            inps,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        generation_args = {}
        generation_args.update(self._generation_args)
        if max_new_tokens:
            generation_args['max_new_tokens'] = max_new_tokens

        # Generate outputs
        outputs = self._model.generate(**inputs, return_dict_in_generate=True, output_scores=return_logprobs, **generation_args)

        # Compute prompt lengths (non-padding tokens per input)
        pad_token_id = self.tokenizer.pad_token_id
        input_ids = inputs["input_ids"]
        prompt_lengths = (input_ids != pad_token_id).sum(dim=1).tolist()  # Count non-pad tokens

        sequences = outputs["sequences"]
        # Remove prompt tokens from generated outputs if needed
        if self._remove_prompt:
            # Only keep tokens generated beyond the prompt length
            sliced_sequences = []
            for i, prompt_length in enumerate(prompt_lengths):
                sliced_sequences.append(sequences[i, prompt_length:])
            sequences = sliced_sequences

        # Decode outputs
        texts = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
        return [
            BackendOutput(text=text)
            for text, length in zip(texts, prompt_lengths)
        ]

    @staticmethod
    def from_params(params: Dict[str, str]) -> 'HuggingFaceBackend':
        """Create a HuggingFaceBackend instance from parameters.

        Supported params:

            - model_id (str): Identifier or path of the HuggingFace model.
            - max_input_length (int): Maximum tokens per input prompt.
            - max_new_tokens (int): Tokens to generate per prompt.
            - logprobs_topk (int): Number of top logprobs to return.
            - verbose (bool): Enable verbose output.

        Returns:
            HuggingFaceBackend: An instance of HuggingFaceBackend.
        """
        return HuggingFaceBackend(
            model_id=params['model_id'],
            max_input_length=int(params.get('max_input_length', 512)),
            max_new_tokens=int(params.get('max_new_tokens', 32)),
            logprobs_topk=int(params.get('logprobs_topk', 20)),
            verbose=params.get('verbose', False) in ['True', 'true', '1'],
        )

    def __repr__(self):
        return f"{self.__class__.__name__}({self.model_id!r})"


class Seq2SeqLMBackend(HuggingFaceBackend):
    _model_class = AutoModelForSeq2SeqLM
    _remove_prompt = False


class StopWordCriteria(StoppingCriteria):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        prompt_size: int,
        stop_words: List[str] = [],
        check_every: int = 1,
    ):
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
        self.max_stop_word_size = max(
            (self.tokenizer.encode(word, return_tensors="pt").size(-1) for word in stop_words),
            default=0,
        )
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
            if any(
                [word in self.tokenizer.decode(latest_tokens, skip_special_tokens=True) for word in self.stop_words]
            ):
                results[i] = True

        return results


__all__ = [
    "HuggingFaceBackend",
    "Seq2SeqLMBackend",
    "StopWordCriteria",
]
