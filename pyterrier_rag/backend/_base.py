import sys
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union

import pyterrier as pt
from more_itertools import chunked
from dataclasses import dataclass


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
        model_name_or_path: model name or checkpoint directory
        supports_logprobs (bool): Flag indicating support for including the logprobs of generated tokens.
        supports_message_input (bool): Flag indicating support for message (chat)-formatted (``List[dict]``) inputs to ``generate``, in addition to ``str`` inputs.
    """
    supports_logprobs = False
    supports_message_input = False

    def __init__(
        self,
        model_name_or_path: str,
        *,
        max_input_length: int = 512,
        max_new_tokens: int = 32,
        verbose: bool = False,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
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
            prompts = [i[self.input_field] for i in inp]
            out = self.backend.generate(prompts, return_logprobs=self.logprobs_field is not None)
            for rec, o in zip(chunk, out):
                result = {**rec, self.output_field: o.text}
                if self.logprobs_field is not None:
                    result[self.logprobs_field] = o.logprobs
                yield result


class _DefaultBackend(Backend):
    def __init__(self):
        self._backend = None

    @property
    def backend(self):
        if self._backend is None:
            raise RuntimeError("You need to run default_backend.set(backend) before using default_backend.")
        return self._backend

    @property
    def supports_logprobs(self):
        return self.backend.supports_logprobs

    @property
    def supports_message_input(self):
        return self.backend.supports_message_input

    @property
    def model_name_or_path(self):
        return self.backend.model_name_or_path

    @property
    def max_input_length(self):
        return self.backend.max_input_length

    @property
    def max_new_tokens(self):
        return self.backend.max_new_tokens

    @property
    def verbose(self):
        return self.backend.verbose

    def set(self, backend: Backend):
        """ Set the default backend to use for text generation.
        
        Parameters:
            backend (Backend): The backend instance to set.
        """
        if self._backend is None:
            sys.stderr.write(f"set default backend to {backend!r}\n")
        else:
            sys.stderr.write(f"replaced default backend {self._backend!r} with {backend!r}\n")
        self._backend = backend
        self.model_name_or_path = backend.model_name_or_path
        self.max_input_length = backend.max_input_length
        self.max_new_tokens = backend.max_new_tokens
        self.verbose = backend.verbose

    def generate(
        self,
        inps: Union[List[str], List[List[dict]]],
        *,
        return_logprobs: bool = False,
        max_new_tokens: Optional[int] = None,
    ) -> List[BackendOutput]:
        """ Delegate the generation to the set backend. """
        return self.backend.generate(inps, return_logprobs=return_logprobs, max_new_tokens=max_new_tokens)

    def __repr__(self):
        if self._backend is None:
            return "<DefaultBackend: not set>"
        return repr(self._backend)


default_backend = _DefaultBackend()


__all__ = ["Backend", "BackendOutput", "TextGenerator", "default_backend"]
