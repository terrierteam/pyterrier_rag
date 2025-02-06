from typing import List, Union, Optional, Any

from ._base import PromptTransformer
from ._context_aggregation import ContextAggregationTransformer


def make_prompt(
        prompt_system_message: str = None,
        prompt_instruction: Union[callable, str] = None,
        model_name_or_path: str = None,
        prompt_conversation_template: str = None,
        prompt_output_field: str = 'query',
        prompt_relevant_fields: List[str] = ['query', 'context'],
        context_in_fields: Optional[List[str]] = ['text'],
        context_out_field: Optional[str] = "context",
        context_intermediate_format: Optional[callable] = None,
        context_tokenizer: Optional[Any] = None,
        context_max_length: Optional[int] = -1,
        context_max_elements: Optional[int] = -1,
        context_max_per_context: Optional[int] = 512,
        truncation_rate: Optional[int] = 50,
        context_aggregate_func: Optional[callable] = None,
        context_per_query: bool = False
):
    prompt = PromptTransformer(
        system_message=prompt_system_message,
        instruction=prompt_instruction,
        model_name_or_path=model_name_or_path,
        conversation_template=prompt_conversation_template,
        output_field=prompt_output_field,
        relevant_fields=prompt_relevant_fields
    )

    context = ContextAggregationTransformer(
        in_fields=context_in_fields,
        out_field=context_out_field,
        intermediate_format=context_intermediate_format,
        tokenizer=context_tokenizer,
        max_length=context_max_length,
        max_elements=context_max_elements,
        max_per_context=context_max_per_context,
        truncation_rate=truncation_rate,
        aggregate_func=context_aggregate_func,
        per_query=context_per_query
    )

    return context >> prompt
