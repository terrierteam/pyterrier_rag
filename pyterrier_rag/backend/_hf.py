from typing import Iterable, List

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
        Backend implementation using HuggingFace Transformer models for text and logit generation.

        Parameters:
            model_name_or_path (str): Identifier or path of the pretrained model.
            model_args (dict): Arguments passed to `from_pretrained` for model instantiation.
            output_format (str): Format for text output (e.g., 'text').
            generation_args (dict): Parameters controlling text generation.
            batch_size (int): Number of inputs to process per batch.
            max_input_length (int): Maximum token length for inputs (defaults to model config).
            max_new_tokens (int): Maximum number of tokens to generate per input.
            verbose (bool): Flag to enable verbose logging.
            **kwargs: Additional keyword arguments passed to `Backend` base class.
        """
    _model_class = AutoModelForCausalLM
    _support_logits = True
    _logit_type = "dense"
    _remove_prompt = True

    def __init__(
        self,
        model_name_or_path: str,
        model_args: dict = {},
        output_format: str = "text",
        generation_args: dict = None,
        batch_size: int = 4,
        max_input_length: int = None,
        max_new_tokens: int = 32,
        return_logits: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(
            output_format=output_format,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            return_logits=return_logits,
            generation_config=None,
            verbose=verbose,
            **kwargs,
        )
        self._model_name_or_path = model_name_or_path
        self._model = (
            None
            if self._model_class is None
            else self._model_class.from_pretrained(model_name_or_path, **model_args).to(self.device).eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self._model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        max_position_embeddings = getattr(self._model.config, "max_position_embeddings", None)
        self.max_input_length = max_input_length or max_position_embeddings

        if generation_args is None:
            generation_args = {
                "max_new_tokens": self.max_new_tokens,
                "temperature": 1.0,
                "do_sample": False,
                "num_beams": 1,
            }
        self._generation_args = generation_args
        self.model = self._model

    @torch.no_grad()
    def generate(self, inps: Iterable[str], **kwargs) -> List[BackendOutput]:
        assert self.model is not None, "Model is not loaded, instantiate a subclass of HFModel"

        # Tokenize inputs
        inputs = self.tokenizer(
            inps,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate outputs
        outputs = self.model.generate(**inputs, return_dict_in_generate=True, output_scores=self.return_logits, **self._generation_args, **kwargs)

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
        if self.return_logits:
            logits = outputs['scores']
            logits = torch.stack(logits, dim=1)
            return [
                BackendOutput(text=text, logits=logits[i], prompt_length=length)
                for i, (text, length) in enumerate(zip(texts, prompt_lengths))
            ]
        else:
            return [
                BackendOutput(text=text, prompt_length=length)
                for text, length in zip(texts, prompt_lengths)
            ]


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
