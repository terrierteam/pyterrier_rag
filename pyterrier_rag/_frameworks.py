from typing import Optional
from functools import partial
from prompts import template

import pandas as pd
import pyterrier as pt
import pyterrier_alpha as pta

from .readers._content_aggregation import dataframe_concat

"""
Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions (IRCOT) ACL 2023

Paper: https://arxiv.org/abs/2212.10509
Implementation Derived from: https://github.com/RUC-NLPIR/FlashRAG/blob/main/flashrag/pipeline/active_pipeline.py#L925
"""

ircot_system_message = 'You serve as an intelligent assistant, adept at facilitating users through complex, multi-hop reasoning across multiple documents. This task is illustrated through demonstrations, each consisting of a document set paired with a relevant question and its multi-hop reasoning thoughts. Your task is to generate one thought for current step, DON\'T generate the whole thoughts at once! If you reach what you believe to be the final step, start with "So the answer is:".'
default_example = "Wikipedia Title: Kurram Garhi\nKurram Garhi is a small village located near the city of Bannu, which is the part of Khyber Pakhtunkhwa province of Pakistan. Its population is approximately 35000. Barren hills are near this village. This village is on the border of Kurram Agency. Other nearby villages are Peppal, Surwangi and Amandi Kala.\n\nWikipedia Title: 2001–02 UEFA Champions League second group stage\nEight winners and eight runners- up from the first group stage were drawn into four groups of four teams, each containing two group winners and two runners- up. Teams from the same country or from the same first round group could not be drawn together. The top two teams in each group advanced to the quarter- finals.\n\nWikipedia Title: Satellite tournament\nA satellite tournament is either a minor tournament or event on a competitive sporting tour or one of a group of such tournaments that form a series played in the same country or region.\n\nWikipedia Title: Trojkrsti\nTrojkrsti is a village in Municipality of Prilep, Republic of Macedonia.\n\nWikipedia Title: Telephone numbers in Ascension Island\nCountry Code:+ 247< br> International Call Prefix: 00 Ascension Island does not share the same country code( +290) with the rest of St Helena.\n\nQuestion: Are both Kurram Garhi and Trojkrsti located in the same country?\nThought: Kurram Garhi is located in the country of Pakistan. Trojkrsti is located in the country of Republic of Macedonia. Thus, they are not in the same country. So the answer is: no.\n\n"


@template
def ircot_prompt(reference: str, question: str) -> str:
    return """{reference}Question: {question}\nThought:"""


@template
def example_format(text: str, title: str = None) -> str:
    return """
    {% if title != None %}
    Wikipedia Title: {{title}}
    {% endif %}
    {{text}}
    """


class IRCOT(pt.Transformer):
    _system_message = ircot_system_message
    _default_example = default_example
    _prompt_template = ircot_prompt
    _example_format = example_format

    def __init__(self,
                 retriever: pt.Transformer,
                 reader: pt.Transformer,
                 system_message: Optional[str] = None,
                 default_example: Optional[str] = None,
                 prompt_template: Optional[template] = None,
                 example_format: Optional[template] = None,
                 context_aggregation: Optional[callable] = None,
                 max_iter: Optional[int] = None,
                 max_docs: Optional[int] = None,
                 ):
        self.retriever = retriever
        self.reader = reader
        self.system_message = system_message or self._system_message
        self.default_example = default_example or self._default_example
        self.prompt_template = prompt_template or self._prompt_template
        self.example_format = example_format or self._example_format
        self.context_aggregation = context_aggregation or partial(dataframe_concat, intermediate_format=self.example_format, relevant_fields=['text', 'title'])
        self.max_iter = max_iter
        self.max_docs = max_docs

        self.is_openai = self.reader.is_openai()

    def _exceeded_max_iter(self, iter: int) -> bool:
        return self.max_iter is not None and iter == self.max_iter

    def exit_condition(self, results: pd.DataFrame) -> bool:
        return "so the answer is" in results.iloc[0].qanswer.lower()

    def construct_prompt(self, reference: str, question: str, previous_gen: str=None) -> str:
        if self.is_openai:
            prompt = []
            prompt.append({
                'role': 'system',
                'content': self.system_message
            }
            )
            prompt.append({
                'role': 'user',
                'content': self.prompt_template(reference=reference, question=question)
            }
            )
            if previous_gen:
                prompt.append({
                    'role': 'system',
                    'content': previous_gen
                }
                )
            return prompt

        else:
            prompt = []
            prompt.append(self.system_message)
            prompt.append(self.prompt_template(reference=reference, question=question))
            if previous_gen:
                prompt.append(previous_gen)
            return '\n'.join(prompt)

    @pta.transform.by_query(add_ranks=False)
    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        previous_thoughts = []
        question = inp.iloc[0].query
        retrieval_results = self.retriever.search(question)

        iter = 1
        stop = False

        while not stop:
            if self.max_docs is not None:
                retrieval_results = retrieval_results.head(self.max_docs)
            reference = self.context_aggregation(retrieval_results)
            prompt = self.construct_prompt(reference, question, ' '.join(previous_thoughts))
            results = self.reader.search(prompt)
            if self.exit_condition(results) or self._exceeded_max_iter(iter):
                stop = True
            current_thought = results.iloc[0].qanswer
            previous_thoughts.append(current_thought)

            current_retrieval_results = self.retriever.search(current_thought)
            # add current retrieval results, overwriting scores if necessary
            retrieval_results = retrieval_results.append(current_retrieval_results, ignore_index=True).sort_values(by='score', ascending=False)
            retrieval_results = retrieval_results.drop_duplicates(subset='docno', keep='last').reset_index(drop=True)

            iter += 1

        return results
