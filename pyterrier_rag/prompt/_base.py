from typing import Optional, Union, Iterable, List, Any, Dict, Tuple, Callable

import re
import jinja2
import pandas as pd
import pyterrier as pt
import pyterrier_alpha as pta
import pyterrier_rag as rag
from pyterrier_rag.prompt.wrapper import prompt
from fastchat.model import get_conversation_template
from fastchat.conversation import get_conv_template


class PromptTransformer(pt.Transformer):
    """
    Transformer that constructs and formats prompts for conversational LLMs.

    Parameters:
        instruction (callable|str): Template or function returning the instruction segment.
        model_name_or_path (str, optional): Model identifier for selecting conversation template.
        system_message (str, optional): System context message for the conversation.
        conversation_template (Any, optional): Preconfigured conversation template.
        api_type (str, optional): API format: 'openai','gemini','vertex','reka'.
        output_field (str): Field name to store the generated prompt.
        input_fields (List[str]): Input record fields required to build the prompt.
        expects_logprobs (bool): Indicator for logprob-based backends.
        answer_extraction (callable, optional): Function to parse model outputs.
        raw_instruction (bool): If True, returns raw instruction without template.
    """
    def __init__(
        self,
        instruction: Union[callable, str] = None,
        model_name_or_path: str = None,
        system_message: Optional[str] = None,
        conversation_template: Optional[Any] = None,
        api_type: Optional[str] = None,
        output_field: str = "prompt",
        input_fields: List[str] = ["query", "qcontext"],
        expects_logprobs: bool = False,
        answer_extraction: Optional[callable] = None,
        raw_instruction: bool = False,
    ):
        self.instruction = instruction
        self.model_name_or_path = model_name_or_path
        self.system_message = system_message
        self.output_field = output_field
        self.input_fields = input_fields
        self.conversation_template = conversation_template
        self.api_type = api_type
        self.expects_logprobs = expects_logprobs
        self.answer_extraction = answer_extraction or self.answer_extraction
        self.raw_instruction = raw_instruction

        self.__post_init__()

    def __post_init__(self):
        if type(self.instruction) is str:
            self.instruction = prompt(self.instruction)
        if self.model_name_or_path is not None:
            self.conversation_template = (
                self.conversation_template or get_conversation_template(self.model_name_or_path)
            )
        if self.conversation_template is None:
            self.conversation_template = get_conv_template("zero_shot")
        if self.system_message is not None:
            # TODO: Set flag for if model supports system message
            self.conversation_template.set_system_message(self.system_message)

        roles = self.conversation_template.roles
        if len(roles) < 2:
            self.user_role, self.llm_role = "user", "assistant"
        else:
            self.user_role = roles[0]
            self.llm_role = roles[1]

        self.output_attribute = (
            {
                "openai": "to_openai_api_messages",
                "gemini": "to_gemini_api_messages",
                "vertex": "to_vertex_api_messages",
                "reka": "to_reka_api_messages",
            }[self.api_type]
            if self.api_type
            else "get_prompt"
        )

    def answer_extraction(self, output):
        return output

    def set_output_attribute(self, supports_message_input: bool):
        # ``output_attribute`` indicates the method to call on the prompt object
        # In the future, we may support more message formats, but for now it's either a string or OpenAI-formatted messages
        self.output_attribute = 'to_openai_api_messages' if supports_message_input else 'get_prompt'

    @property
    def prompt(self):
        return self.conversation_template.copy()

    def to_output(self, prompt) -> Union[str, List[dict]]:
        return getattr(prompt, self.output_attribute)()

    def create_prompt(self, fields: dict) -> Union[str, List[dict]]:
        current_prompt = self.prompt
        instruction = self.instruction(**fields)
        if self.raw_instruction:
            return instruction
        current_prompt.append_message(self.user_role, instruction)
        return self.to_output(current_prompt)

    @pta.transform.by_query(add_ranks=False)
    def transform_iter(self, inp: Iterable[dict]) -> Iterable[dict]:
        return self.transform_by_query(inp)

    def transform_by_query(self, inp: Iterable[dict]) -> Iterable[dict]:
        inp = list(inp)
        qid = inp[0].get("qid", None)
        query = inp[0].get("query", None)
        fields = {k: v for k, v in inp[0].items() if k in self.input_fields}
        if any([f not in fields for f in self.input_fields]):
            message = f"Expected {self.input_fields} but recieved {fields}"
            raise ValueError(message)
        prompt = self.create_prompt(fields)
        return [{self.output_field: prompt, "qid": qid, "query_0": query}]


class Prompt(pt.Transformer):
    def __init__(self, *, backend: Optional[rag.Backend] = None, group_by: Optional[Tuple[str]] = None):
        self.backend = backend or rag.default_backend
        self.group_by = group_by

    def build_prompt(self, inp: Dict[str, Any]) -> Union[str, List[Dict[str, str]]]:
        return inp['prompt']

    def generate(self, prompts: Union[str, List[Dict[str, str]]]) -> rag.BackendOutput:
        return self.backend.generate(prompts)

    def extract_output(self, output: rag.BackendOutput) -> Union[str, Dict[str, Any]]:
        return output.text

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        prompts = []
        results = []
        if self.group_by: # grouped records
            for keys, group in inp.groupby(self.group_by):
                rec = dict(zip(self.group_by, keys))
                prompts.append(self.backend.build_prompt({**rec, 'group': group.to_dict(orient='records')}))
                results.append(rec)
        else: # record-by-record
            for row in inp.itertuples(index=False):
                rec = row._asdict()
                prompts.append(self.backend.build_prompt(rec))
                results.append(rec)

        outputs = self.generate(prompts)
        for res, output in zip(results, outputs):
            ext_output = self.extract_output(output)
            if isinstance(ext_output, str):
                res['qanswer'] = ext_output
            else:
                res.update(ext_output)
        return pd.DataFrame(results)


class JinjaPromptMixin:
    """Mixin for building prompts using Jinja2 templates.

    This mixin allows you to define a Jinja2 template for the prompt, which can be rendered with the input data.
    The template should be defined in the ``jinja_template`` attribute.
    """
    prompt_template = '{{prompt}}'
    def build_prompt(self, inp: Dict[str, Any]) -> Union[str, List[Dict[str, str]]]:
        if not hasattr(self, '_prompt_template'):
            self._prompt_template = jinja2.Template(self.prompt_template)
        return self._prompt_template.render(**inp)


class RegexOutputMixin:
    """Mixin for extracting output using a regular expression.

    This is useful for parsing structured outputs from LLMs, e.g., when prompted to output a response in a specific format.

    The ``output_regex`` attribute should be set to the desired regex pattern.

    The extraction will match the first occurrence of the regex pattern in the output text and output the following:

     - If there are any named capture groups (e.g., ``Relevance: (?P<relevance>\\d+)``), values are extracted and
       returned as a dictionary with keys corresponding to the group names. (When no match is found, all group names
       are returned with empty strings.)
     - If there are capture groups but none of them are named, the first capture group is returned as ``qanswer``.
     - If there are no capture groups (or they are all marked as non-capturing groups, e.g., ``(?:\\w)``), the entire
       match is returned as ``qanswer``.

    When named capture groups are used, you may optinally provide ``output_field_types`` as a dictionary mapping
    group names to functions that cast the extracted values to the desired type. For example, if you have a group
    named ``age`` and you want to convert it to an integer, you can set ``output_field_types = {'relevance': int}``.
    In the case where a mattern or capture group fails to match, the value is set to the default value of the type
    function (e.g., ``int()`` returns ``0``, ``str()`` returns an empty string, etc.).
    """
    output_regex = ''
    output_field_types: Dict[str, Callable[[str], Any]] = {}
    def extract_output(self, output: rag.BackendOutput) -> Union[str, Dict[str, Any]]:
        if not self.output_regex:
            return super().extract_output(output)
        if not hasattr(self, '_output_regex'):
            self._output_regex = re.compile(self.output_regex)

        match = self._output_regex.search(output.text)
        if not match:
            if len(self._output_regex.groupindex) > 0:
                return {k: self._regex_output_cast(k) for k in self._output_regex.groupindex.keys()}
            return ''

        if len(self._output_regex.groupindex) > 0:
            return {k: self._regex_output_cast(k, v) for k, v in match.groupdict().items()}
        if len(match.groups()) == 0:
            return match.group(0)
        return match.group(1)

    def _regex_output_cast(self, key: str, value: Optional[str] = None) -> Any:
        if key in self.output_field_types:
            if value is None:
                return self.output_field_types[key]()
            return self.output_field_types[key](value)
        return value


class SystemPromptMixin:
    """Mixin for setting a system message in the prompt.

    This mixin allows you to set a system message that will be included in the prompt.
    The system message is typically used to provide context or instructions to the LLM.
    """
    system_prompt: Optional[str] = None
    system_prompt_role: str = 'developer'

    def build_prompt(self, inp: Dict[str, Any]) -> Union[str, List[Dict[str, str]]]:
        result = super().build_prompt(inp)
        if not self.system_prompt:
            return result
        if isinstance(result, str):
            if self.backend.supports_message_input:
                return [
                    {'role': self.system_prompt_role, 'message': self.system_prompt},
                    {'role': 'user', 'message': result},
                ]
            else:
                return f'{self.system_prompt_role}: {self.system_prompt}\n\n{result}'
        else:
            return [
                {'role': self.system_prompt_role, 'message': self.system_prompt},
                *result
            ]


class BasicPrompt(SystemPromptMixin, JinjaPromptMixin, RegexOutputMixin, Prompt):
    """Basic prompt transformer that uses Jinja2 templates, an optional system prompt, and regex for output extraction.

    This class combines the functionality of building prompts using Jinja2 templates and extracting structured
    outputs using regular expressions. It is useful for cases where you want to format prompts dynamically and
    parse structured responses from LLMs.

    See :class:`pyterrier_rag.JinjaPromptMixin`, :class:`pyterrier_rag.RegexOutputMixin`, and
    :class:`pyterrier_rag.SystemPromptMixin` for more usage details.
    """
    jinja_template = '{{prompt}}'
    output_regex = ''
    output_field_types: Dict[str, Callable[[str], Any]] = {}
    system_prompt: Optional[str] = None
    system_prompt_role: str = 'developer'


__all__ = ["PromptTransformer", "Prompt", "JinjaPromptMixin", "RegexOutputMixin", "SystemPromptMixin", "BasicPrompt"]
