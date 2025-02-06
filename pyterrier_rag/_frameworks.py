from typing import Any, List, Optional, Union
from outlines import prompt

import pandas as pd
import pyterrier as pt
import pyterrier_alpha as pta

from .prompt import make_prompt


class Iterative(pt.Transformer):
    def __init__(
        self,
        pipeline: pt.Transformer,
        exit_condition: callable = lambda _: False,
        max_iter: Optional[int] = None,
    ):
        self.pipeline = pipeline
        self.exit_condition = exit_condition
        self.max_iter = max_iter

    def _exceeded_max_iter(self, iter: int) -> bool:
        return self.max_iter is not None and iter == self.max_iter

    @pta.transform.by_query(add_ranks=False)
    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        iter = 1
        stop = False
        while not stop:
            inp = self.pipeline.transform(inp)
            if self.exit_condition(inp) or self._exceeded_max_iter(iter):
                stop = True
        return inp


"""
Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions (IRCOT) ACL 2023

Paper: https://arxiv.org/abs/2212.10509
Implementation Derived from: https://github.com/RUC-NLPIR/FlashRAG/blob/main/flashrag/pipeline/active_pipeline.py#L925
"""

ircot_system_message = 'You serve as an intelligent assistant, adept at facilitating users through complex, multi-hop reasoning across multiple documents. This task is illustrated through demonstrations, each consisting of a document set paired with a relevant question and its multi-hop reasoning thoughts. Your task is to generate one thought for current step, DON\'T generate the whole thoughts at once! If you reach what you believe to be the final step, start with "So the answer is:".'


@prompt
def ircot_prompt(context: str, query: str) -> str:
    """{context}Question: {query}\nThought:"""


@prompt
def ircot_example_format(text: str, title: str = None) -> str:
    """
    {% if title != None %}
    Wikipedia Title: {{title}}
    {% endif %}
    {{text}}
    """


class ReScorerTransformer(pt.Transformer):
    def __init__(self):
        super().__init__()
        self.scores = {}
        self.texts = {}

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        scores = inp.set_index(['query_id', 'doc_id'])['score'].to_dict()
        texts = inp.set_index(['query_id', 'doc_id'])['text'].to_dict()

        self.scores.update(scores)
        self.texts.update(texts)

        current_query_ids = inp['query_id'].unique().tolist()
        query_ids = [k[0] for k in self.scores.keys() if k[0] in current_query_ids]

        doc_ids = [k[1] for k in self.scores.keys() if k[0] in current_query_ids]
        scores = [v for k, v in self.scores.items() if k[0] in current_query_ids]
        texts = [self.texts[(query_id, doc_id)] for query_id, doc_id in zip(query_ids, doc_ids)]

        return pd.DataFrame({
            'qid': query_ids,
            'docno': doc_ids,
            'score': scores,
            'text': texts
        }).sort_values(['qid', 'score'], ascending=[True, False])


class IRCOT(Iterative):

    def __init__(
        self,
        retriever: pt.Transformer,
        reader: pt.Transformer,
        max_iter: Optional[int] = None,
        max_docs: Optional[int] = None,
        exit_condition: callable = lambda x: "so the answer is" in x.iloc[0]['query'].lower(),
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
        self.retriever = retriever
        self.reader = reader
        _system_message = prompt_system_message or ircot_system_message
        _prompt_instruction = prompt_instruction or ircot_prompt
        _example_format = context_intermediate_format or ircot_example_format

        self.prompt = make_prompt(
            prompt_system_message=_system_message,
            prompt_instruction=_prompt_instruction,
            model_name_or_path=model_name_or_path,
            prompt_conversation_template=prompt_conversation_template,
            prompt_output_field=prompt_output_field,
            prompt_relevant_fields=prompt_relevant_fields,
            context_in_fields=context_in_fields,
            context_out_field=context_out_field,
            context_intermediate_format=_example_format,
            context_tokenizer=context_tokenizer,
            context_max_length=context_max_length,
            context_max_elements=context_max_elements,
            context_max_per_context=context_max_per_context,
            context_truncation_rate=truncation_rate,
            context_aggregate_func=context_aggregate_func,
            context_per_query=context_per_query
        )

        self.max_docs = max_docs

        super().__init__(
            pipeline=self.retriever >> ReScorerTransformer() >> self.prompt >> self.reader,
            exit_condition=exit_condition,
            max_iter=max_iter
        )
