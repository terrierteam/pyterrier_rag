from dataclasses import dataclass, field
from typing import List, Union, Optional, Any


@dataclass
class PromptConfig:
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "The model checkpoint or endpoint for formatting"},
    )
    system_message: Optional[str] = field(
        default=None, metadata={"help": "The system message to use for the prompt"}
    )
    instruction: Optional[Union[callable, str]] = field(
        default=None, metadata={"help": "The instruction to use for the prompt"}
    )
    conversation_template: Optional[Any] = field(
        default=None,
        metadata={"help": "The conversation template to use for the prompt"},
    )
    output_field: Optional[str] = field(
        default="query", metadata={"help": "The output field for the prompt"}
    )
    input_fields: Optional[List[str]] = field(
        default_factory=lambda: ["query", "context"],
        metadata={"help": "The input fields for the prompt"},
    )
    api_type: Optional[str] = field(
        default=None, metadata={"help": "The API type for the prompt"}
    )

    def __post_init__(self):
        assert (
            self.model_name_or_path or self.conversation_template
        ), "Either model_name_or_path or conversation_template must be provided"
        self.instruction = (
            self.instruction
            if isinstance(self.instruction, callable)
            else self.instruction.format
        )


@dataclass
class ContextConfig:
    in_fields: Optional[List[str]] = field(
        default_factory=lambda: ["text"],
        metadata={"help": "The input fields for the context"},
    )
    out_field: Optional[str] = field(
        default="context", metadata={"help": "The output field for the context"}
    )
    intermediate_format: Optional[callable] = field(
        default=None, metadata={"help": "The intermediate format for the context"}
    )
    tokenizer: Optional[Any] = field(
        default=None, metadata={"help": "The tokenizer for the context"}
    )
    max_length: Optional[int] = field(
        default=-1, metadata={"help": "The maximum length for the context"}
    )
    max_elements: Optional[int] = field(
        default=-1, metadata={"help": "The maximum  elements for the context"}
    )
    max_per_context: Optional[int] = field(
        default=512, metadata={"help": "The maximum per context for the context"}
    )
    truncation_rate: Optional[int] = field(
        default=50, metadata={"help": "The truncation rate for the context"}
    )
    aggregate_func: Optional[callable] = field(
        default=None, metadata={"help": "The aggregate function for the context"}
    )
    per_query: Optional[bool] = field(
        default=False, metadata={"help": "The per query for the context"}
    )
