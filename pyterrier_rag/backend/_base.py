from abc import ABC, abstractmethod
from typing import List, Dict, Optional

import pyterrier as pt
from more_itertools import chunked
from dataclasses import dataclass


@dataclass
class BackendOutput:
    text: str = None
    logits: List[Dict[str, float]] = None
    prompt_length: int = None


class Backend(pt.Transformer, ABC):
    """
    Abstract base class for model-backed Transformers in PyTerrier.

    Subclasses must implement the raw generation logic (generate) and the
    high-level generate method. Supports optional logit extraction and prompt
    trimming.

    Parameters:
        max_input_length (int): Maximum token length for each input prompt.
        max_new_tokens (int): Maximum number of tokens to generate.
        verbose (bool): Flag to enable detailed logging.
        device (Union[str, torch.device]): Device for model execution.
    Attributes:
        model_name_or_path: model name or checkpoint directory
        support_logits (bool): Flag indicating logit support.
        _api_type (str): If using API do not return string
    """

    support_logits = False
    _api_type = None

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
    def generate(self,
        inps: List[str],
        *,
        return_logits: bool = False,
        max_new_tokens: Optional[int] = None,
    ) -> List[BackendOutput]:
        """ Generate text from input prompts.

        Parameters:
            inps (List[str]): Input prompts to generate text for.
            return_logits (bool): Whether to return logits along with text. (Only available if ``backend.support_logits==True``.)
            max_new_tokens (Optional[int]): Override for max tokens to generate.

        Returns:
            List[BackendOutput]: An output for each ``inp``, each containing the generated text and optionally logits.
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

    def logit_generator(self,
        *,
        input_field: str = 'prompt',
        output_field: str = 'qanswer',
        logits_field: str = 'qanswer_logits',
        batch_size: int = 4,
        max_new_tokens: Optional[int] = None,
    ) -> pt.Transformer:
        """ Create a text generator transformer that also returns the logits of each token using this backend.

        Parameters:
            input_field (str): Name of the field containing input prompts.
            output_field (str): Name of the field to store generated text.
            logits_field (str): Name of the field to store logits.
            batch_size (int): Number of prompts to process in each batch.
            max_new_tokens (Optional[int]): Override for max tokens to generate. If None, uses the backend's max_new_tokens.
        """
        if not self.support_logits:
            raise ValueError("This model cannot return logits")
        return TextGenerator(self, input_field=input_field, output_field=output_field, logits_field=logits_field, max_new_tokens=max_new_tokens)

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
        logits_field: Optional[str] = None,
        batch_size: int = 4,
        max_new_tokens: Optional[int] = None,
    ):
        """
        Parameters:
            backend (Backend): The backend to use for text generation.
            input_field (str): Name of the field containing input prompts.
            output_field (str): Name of the field to store generated text.
            logits_field (Optional[str]): Name of the field to store logits. If None, logits are not returned.
            batch_size (int): Number of prompts to process in each batch.
            max_new_tokens (Optional[int]): Override for max tokens to generate. If None, uses the backend's max_new_tokens.
        """
        if logits_field is not None and not backend.support_logits:
            raise ValueError("Backend does not support logits")
        self.backend = backend
        self.input_field = input_field
        self.output_field = output_field
        self.logits_field = logits_field
        self.batch_size = batch_size

    def transform_iter(self, inp: pt.model.IterDict) -> pt.model.IterDict:
        for chunk in chunked(inp, self.batch_size):
            chunk = list(chunk)
            prompts = [i[self.input_field] for i in inp]
            out = self.backend.generate(prompts, return_logits=self.logits_field is not None)
            for rec, o in zip(chunk, out):
                result = {**rec, self.output_field: o.text}
                if self.logits_field is not None:
                    result[self.logits_field] = o.logits
                yield result


__all__ = ["Backend", "BackendOutput", "TextGenerator"]
