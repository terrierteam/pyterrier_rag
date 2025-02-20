from typing import Optional, List, Union, Literal
import random
import itertools
from functools import partial
from collections import Counter
from outlines import prompt

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


@prompt
def ircot_prompt(reference: str, question: str) -> str:
    """{reference}Question: {question}\nThought:"""


@prompt
def example_format(text: str, title: str = None) -> str:
    """
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
                 prompt_template: Optional[prompt] = None,
                 example_format: Optional[prompt] = None,
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


class Iterative(pt.Transformer):

    def __init__(self, retriever : pt.Transformer, reader : pt.Transformer, max_iter : Optional[int] =None):
        self.retriever = retriever
        self.reader = reader
        self.max_iter = max_iter

    @pta.transform.by_query(add_ranks=False)
    def transform(self, inp : pd.DataFrame) -> pd.DataFrame:
        iter = 1
        stop = False
        while not stop:
            results = self.retriever(inp)
            answers = self.reader(results)
            # TODO is self.reader assumed to append LLM output to query?
            if self.max_iter is not None and iter == self.max_iter:
                stop = True
            # TODO should be more customisable - perhaps a lambda?
            if "the answer is" in answers.iloc[0].qanswer.lower():
                stop = True
            inp = answers
        return answers


class Genetic(pt.Transformer):
    """Genetic RAG pipeline (Gen2IR)

    .. cite.dblp:: conf/doceng/KulkarniYGFM23
    """

    def __init__(self,
        fitness: pt.Transformer,
        mutators: List[pt.Transformer],
        *,
        convergence_depth: int = 2,
        mutation_depth: int = 2,
        mutations_per_generation: int = 8,
        response_type: Union[Literal['result_frame'], Literal['answer_frame']] = 'answer_frame',
        rng: Optional[int] = None,
    ):
        """
        Args:
            fitness: a Transformer that scores the input DataFrame (to determine the best answers)
            mutators: a list of Transformers that generate new answers
            convergence_depth: the depth at which we consider the results to have converged
            mutation_depth: the depth at which sample from for mutations
            mutations_per_generation: the number of mutations to generate per generation
            response_type: the type of frame to return: either an ``answer_frame`` (which includes only a single qanswer per query)
                or a ``result_frame`` which includes all retrieved generated documents.
            rng: the random seed
        """
        self.fitness = fitness
        self.mutators = mutators
        self.convergence_depth = convergence_depth
        self.mutation_depth = mutation_depth
        self.mutations_per_generation = mutations_per_generation
        self.response_type = response_type
        self.rng = random.Random(rng) or random.Random()

    @pta.transform.by_query(add_ranks=False)
    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        pta.validate.result_frame(inp, extra_columns=['query', 'text'])
        qid, query = inp['qid'].iloc[0], inp['query'].iloc[0]
        scores = Counter()
        threshold = float('-inf')
        next_res = inp
        for generation in itertools.count():
            # Evaluate Fitness
            next_res = self.fitness(next_res)
            for docno, text, score in next_res[['docno', 'text', 'score']].itertuples(index=False):
                scores[docno, text] = score
            sorted_res = scores.most_common()
            if len(next_res[next_res['score'] > threshold]) == 0:
                break # converged
            threshold = sorted_res[self.convergence_depth][1]

            # Mutate
            top_frame = pd.DataFrame({
                'qid': qid,
                'query': query,
                'docno': [r[0][0] for r in sorted_res[:self.mutation_depth]],
                'text': [r[0][1] for r in sorted_res[:self.mutation_depth]],
                'score': [r[1] for r in sorted_res[:self.mutation_depth]],
                'rank': list(range(len(sorted_res[:self.mutation_depth]))),
            })
            next_res = []
            for i in range(self.mutations_per_generation):
                mutator = self.rng.choice(self.mutators)
                answer = mutator(top_frame)
                assert len(answer) == 1
                next_res.append({
                    'qid': qid,
                    'query': query,
                    'docno': f'g{generation}i{i}',
                    'text': answer['qanswer'].iloc[0],
                })
            next_res = pd.DataFrame(next_res)

        if self.response_type == 'answer_frame':
            return pd.DataFrame({
                'qid': [qid],
                'query': [query],
                'qanswer': [sorted_res[0][0][1]],
            })
        elif self.response_type == 'result_frame':
            return pd.DataFrame({
                'qid': qid,
                'query': query,
                'docno': [r[0][0] for r in sorted_res],
                'text': [r[0][1] for r in sorted_res],
                'score': [r[1] for r in sorted_res],
                'rank': list(range(len(sorted_res))),
            })
        else:
            raise ValueError(f'unknown response_type: {self.response_type!r}')
