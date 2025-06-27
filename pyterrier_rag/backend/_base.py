import re
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union

import pyterrier as pt
from more_itertools import chunked
from dataclasses import dataclass

import pyterrier_rag as ptr
import pyterrier_rag.backend


@dataclass
class BackendOutput:
    text: str = None
    logprobs: Optional[List[Dict[str, float]]] = None


class Backend(pt.Transformer, ABC):
    """
    Abstract base class for model-backed Transformers in PyTerrier.

    Subclasses must implement the raw generation logic (generate) and the
    high-level generate method. Supports optional logprob extraction.

    Parameters:
        max_input_length (int): Maximum token length for each input prompt.
        max_new_tokens (int): Maximum number of tokens to generate.
        verbose (bool): Flag to enable detailed logging.
        device (Union[str, torch.device]): Device for model execution.

    Attributes:
        model_id: model name or checkpoint path
        supports_logprobs (bool): Flag indicating support for including the logprobs of generated tokens.
        supports_message_input (bool): Flag indicating support for message (chat)-formatted (``List[dict]``) inputs to ``generate``, in addition to ``str`` inputs.
    """
    supports_logprobs = False
    supports_message_input = False
    supports_num_responses = False

    def __init__(
        self,
        model_id: str,
        *,
        max_input_length: int = 512,
        max_new_tokens: int = 32,
        verbose: bool = False,
    ):
        super().__init__()
        self.model_id = model_id
        self.max_input_length = max_input_length
        self.max_new_tokens = max_new_tokens
        self.verbose = verbose

    # Abstract methods

    @abstractmethod
    def generate(
        self,
        inps: Union[List[str], List[List[dict]]],
        *,
        return_logprobs: bool = False,
        max_new_tokens: Optional[int] = None,
    ) -> List[BackendOutput]:
        """ Generate text from input prompts.

        Parameters:
            inps (Union[List[str], List[List[dict]]]): Input prompts as strings or dictionaries. When strings, represent the prompts directly. When a list of dictionaries, represents a sequence of messages (if ``backend.supports_message_input==True``).
            return_logprobs (bool): Whether to return logprobs of generated tokens along with text. (Only available if ``backend.supports_logprobs==True``.)
            max_new_tokens (Optional[int]): Override for max tokens to generate.

        Returns:
            List[BackendOutput]: An output for each ``inp``, each containing the generated text and optionally logprobs.
        """
        raise NotImplementedError("Implement the generate method")

    # Transformer implementations

    def text_generator(self,
        *,
        input_field: str = 'prompt',
        output_field: str = 'qanswer',
        batch_size: int = 4,
        max_new_tokens: Optional[int] = None,
    ) -> pt.Transformer:
        """ Create a text generator transformer using this backend.

        Parameters:
            input_field (str): Name of the field containing input prompts.
            output_field (str): Name of the field to store generated text.
            batch_size (int): Number of prompts to process in each batch.
            max_new_tokens (Optional[int]): Override for max tokens to generate. If None, uses the backend's max_new_tokens.
        """
        return TextGenerator(self, input_field=input_field, output_field=output_field, max_new_tokens=max_new_tokens)

    def logprobs_generator(self,
        *,
        input_field: str = 'prompt',
        output_field: str = 'qanswer',
        logprobs_field: str = 'qanswer_logprobs',
        batch_size: int = 4,
        max_new_tokens: Optional[int] = None,
    ) -> pt.Transformer:
        """ Create a text generator transformer that also returns the logprobs of each token using this backend.

        Parameters:
            input_field (str): Name of the field containing input prompts.
            output_field (str): Name of the field to store generated text.
            logprobs_field (str): Name of the field to store logprobs.
            batch_size (int): Number of prompts to process in each batch.
            max_new_tokens (Optional[int]): Override for max tokens to generate. If None, uses the backend's max_new_tokens.
        """
        if not self.supports_logprobs:
            raise ValueError("This model cannot return logprobs")
        return TextGenerator(self, input_field=input_field, output_field=output_field, logprobs_field=logprobs_field, max_new_tokens=max_new_tokens)

    def transform_iter(self, inp: pt.model.IterDict) -> pt.model.IterDict:
        return self.text_generator().transform_iter(inp)

    # factory methods

    @staticmethod
    def from_dsn(dsn: str) -> 'Backend':
        """ Create a Backend instance from a DSN (Data Source Name) string.

        The DSN format is: ``<provider>:<model_id> [key1=value1 key2=value2 ...]``.

        Examples: ``"openai:gpt-3.5-turbo"``, ``"openai:meta-llama/Llama-4-Scout-17B-16E-Instruct base_path=http://localhost:8080/v1"``,
        ``"vllm:meta-llama/Llama-4-Scout-17B-16E-Instruct"``, ands ``"huggingface:meta-llama/Llama-4-Scout-17B-16E-Instruct"``.

        See each backend implementation ``from_params`` method for their supported keys.

        Parameters:
            dsn (str): The DSN string to parse.

        Returns:
            Backend: An instance of the appropriate Backend subclass based on the provider.

        Raises:
            ValueError: If the DSN format is invalid or the provider is unknown.
        """
        pattern = r"^(?P<backend>\w+):(?P<model_id>[\w/-]+)(?:\s+(?P<params>.*))?$"
        match = re.match(pattern, dsn)
        if not match:
            raise ValueError(f"Invalid DSN format: {dsn!r}")

        backend = match.group("backend")
        backend_cls = {
            'openai': pyterrier_rag.backend.OpenAIBackend,
            'huggingface': pyterrier_rag.backend.HuggingFaceBackend,
            'vllm': pyterrier_rag.backend.VLLMBackend,
        }.get(backend)
        if backend_cls is None:
            raise ValueError(f'unknown backend {backend}')

        params_str = match.group("params")
        params = {
            'model_id': match.group("model_id"),
        }
        if params_str:
            for param in params_str.split():
                key, value = param.split("=")
                params[key] = value
        return backend_cls.from_params(params)


class TextGenerator(pt.Transformer):
    """ Transformer that generates text from the specified backend.
    """
    def __init__(self,
        backend: Backend,
        *,
        input_field: str = 'prompt',
        output_field: str = 'qanswer',
        logprobs_field: Optional[str] = None,
        batch_size: int = 4,
        max_new_tokens: Optional[int] = None,
    ):
        """
        Parameters:
            backend (Backend): The backend to use for text generation.
            input_field (str): Name of the field containing input prompts.
            output_field (str): Name of the field to store generated text.
            logprobs_field (Optional[str]): Name of the field to store generated logprobs. If None, logprobs are not returned.
            batch_size (int): Number of prompts to process in each batch.
            max_new_tokens (Optional[int]): Override for max tokens to generate. If None, uses the backend's max_new_tokens.
        """
        if logprobs_field is not None and not backend.supports_logprobs:
            raise ValueError("Backend does not support logprobs")
        self.backend = backend
        self.input_field = input_field
        self.output_field = output_field
        self.logprobs_field = logprobs_field
        self.batch_size = batch_size

    def transform_iter(self, inp: pt.model.IterDict) -> pt.model.IterDict:
        for chunk in chunked(inp, self.batch_size):
            chunk = list(chunk)
            prompts = [i[self.input_field] for i in chunk]
            out = self.backend.generate(prompts, return_logprobs=self.logprobs_field is not None)
            for rec, o in zip(chunk, out):
                result = {**rec, self.output_field: o.text}
                if self.logprobs_field is not None:
                    result[self.logprobs_field] = o.logprobs
                yield result


__all__ = ["Backend", "BackendOutput", "TextGenerator"]
