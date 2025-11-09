import re
from typing import List, Dict, Any, Optional

import pyterrier as pt
import pandas as pd

from ... import Backend

class AgenticRAG(pt.Transformer):

    def __init__(
        self,
        retriever: pt.Transformer,
        backend: Backend,
        start_search_tag: str,
        end_search_tag: str,
        start_results_tag: str,
        end_results_tag: str,
        start_answer_tag: str,
        end_answer_tag: str,
        prompt_template: str = "{question}",
        top_k: int = 5,
        max_turn: int = 10,
        stop_sequences: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.retriever = retriever
        self.backend = backend
        self.prompt_template = prompt_template
        self.top_k = top_k
        self.max_turn = max_turn

        # implement in subclasses
        self.start_search_tag = start_search_tag
        self.end_search_tag = end_search_tag
        self.start_results_tag = start_results_tag
        self.end_results_tag = end_results_tag
        self.start_answer_tag = start_answer_tag
        self.end_answer_tag = end_answer_tag

        self.stop_sequences = stop_sequences if stop_sequences else [self.end_search_tag, self.end_answer_tag]

    def generate(self, context: List[str]) -> List[str]:
        return [
            t.text
            for t in self.backend.generate(
                context,
                stop_sequences=self.stop_sequences,
            )
        ]

    def get_prompt(self, question: str) -> str:
        return self.prompt_template.format(question=question) if self.prompt_template else question

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        state_active_queries : List[Dict[str,Any]] = []
        for _, row in df.iterrows():
            state = {
                'qid': str(row['qid']),
                'query': row['query'],
                'context': self.get_prompt(row["query"]),
                'search_history': [],
                'search_iterations' : 0,
                'qanswer' : '',
                'output' : '',
                'stop_reason' : None,
            }
            state_active_queries.append(state)
        state_finished_queries : List[Dict[str,Any]] = []

        for turn in range(self.max_turn):
            if not state_active_queries:
                break

            # 1. call the LLM for each query still active
            outputs : List[str] = self.generate([q['context'] for q in state_active_queries])  # outputs: List[str]

            # 2. check for answer in each of the :
            # if we see the question has been answered:
            # extract the answer
            # remove this query from state_active, add to state_finished list

            batch_answers : List[str|None] = self.check_answers(outputs)  # List[answer or None]
            # one for every active query
            assert len(batch_answers) == len(state_active_queries)
            pending_queries : List[Dict[str,Any]] = []

            for i, answer in enumerate(batch_answers):
                this_query_state = state_active_queries[i]
                this_query_state["context"] += "\n\n" + outputs[i]
                this_query_state['output'] += "\n\n" + outputs[i]

                # 1) determine whether there is an answer (to avoid misjudging the first sentence as the answer)
                if answer is not None:
                    this_query_state['qanswer'] = answer
                    this_query_state['stop_reason'] = 'Got answer'
                    if not outputs[i].endswith(self.end_answer_tag):
                        this_query_state["context"] += self.end_answer_tag
                        this_query_state["output"] += self.end_answer_tag
                    state_finished_queries.append(this_query_state)
                    continue

                # 2) extract the search query, terminate if no query found
                next_search = self.extract_search_query(outputs[i])
                if not next_search:
                    this_query_state['stop_reason'] = 'No answer, no search'
                    state_finished_queries.append(this_query_state)
                    continue

                # restore stop sequences
                if self.end_search_tag and not outputs[i].endswith(self.end_search_tag):
                    this_query_state["context"] += self.end_search_tag
                    this_query_state["output"] += self.end_search_tag

                # record query for next stage
                this_query_state['search_history'].append(next_search)
                this_query_state['search_iterations'] += 1
                pending_queries.append(this_query_state)
            state_active_queries = pending_queries

            if len(pending_queries) == 0:
                break

            # 4. exectute queries
            batch_queries = pd.DataFrame({
                "qid": [ f"{q['qid']}-{len(q['search_history'])}"  for q in pending_queries ],
                "query": [ q['search_history'][-1] for q in pending_queries ]
            })

            # batch_queries["qid"] = batch_queries["qid"].astype(str)
            batch_results = (self.retriever % self.top_k).transform(batch_queries)

            # 5. replace state_active_queries with pending_queries
            next_state_active_queries = []
            for i, q in enumerate(pending_queries):
                batch_results["qid"] = batch_results["qid"].astype(str)
                this_q_results = batch_results[batch_results.qid.str.startswith(q['qid'] + "-")]
                if len(this_q_results):
                    q["context"] += self.wrap_search_results(this_q_results)
                    next_state_active_queries.append(q)
                else:
                    q['stop_reason'] = 'No retrieval results'
                    state_finished_queries.append(q)
                    continue

            state_active_queries = next_state_active_queries

        # any still active queries must have had no answer after self.max_turns
        if state_active_queries:
            for q in state_active_queries:
                if not q.get("stop_reason"):
                    q["stop_reason"] = "No answer after max turns"

        # 7. combine state_finished into results_df, and anything left in state_active that
        results = state_finished_queries + state_active_queries
        return pd.DataFrame(results)

    def check_answers(self, model_outputs: List[str]) -> List[str]:
        results = []
        for output in model_outputs:
            if self.start_answer_tag in output:
                answer = output.split(self.start_answer_tag,1)[1].strip()
                results.append(answer)
            else:
                results.append(None)
            # answer = self.format_answers(output, strict=True)
            # if answer == "no answer found":
            #     results.append(None)
            # else:
            #     results.append(answer)
        return results

    # get search query from the output, can be similar among different models
    def extract_search_query(self, output: str) -> Optional[str]:
        if output is None:
            return None
        start_tag = self.start_search_tag
        end_tag = self.end_search_tag

        start_idx = output.find(start_tag)
        if start_idx == -1:
            return None
        start_idx += len(start_tag)
        end_idx = output.find(end_tag, start_idx)
        if end_idx == -1:
            end_idx = len(output)

        query = output[start_idx:end_idx].strip()
        if not query:
            return None

        query = (
            query.replace('"', "")
                 .replace("'", "")
                 .replace("\t", " ")
                 .replace("...", "")
                 .strip()
        )

        # normalise whitespace
        query = re.sub(r"\s+", " ", query) if query else ""
        return query if query else None

    def wrap_search_results(self, docs: pd.DataFrame) -> str:
        if docs is None or len(docs) == 0:
            return f"{self.start_results_tag}{self.end_results_tag}"

        # Prioritize common text columns
        for col in ["text", "body", "raw", "contents", "title"]:
            if col in docs.columns:
                docs_str = "\n".join(docs[col].astype(str).tolist())
                return f"{self.start_results_tag}{docs_str}{self.end_results_tag}"

        # When there is no text column, create some readable fields for easy troubleshooting
        meta_cols = [c for c in ["docno", "docid", "rank", "score"] if c in docs.columns]
        if meta_cols:
            docs_str = "\n".join(
                docs[meta_cols].astype(str).agg(" | ".join, axis=1).tolist()
            )
            return f"{self.start_results_tag}{docs_str}{self.end_results_tag}"

        return f"{self.start_results_tag}{self.end_results_tag}"
        # If there is no string, return the entire line.
        # return "\n".join(docs.astype(str).agg(" | ".join, axis=1).tolist())

    def format_answers(self, output: str, strict: bool = False) -> str:
        # 1) 显式 <answer> 标签
        match = re.search(r"<answer>(.*?)</answer>", output, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            if answer:
                return answer

        # 2) LaTeX 风格 \\boxed{...}（可含 \\text{...}）
        match = re.search(r"\\boxed\{(.*)\}", output, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            if answer:
                inner_match = re.search(r"\\text\{(.*)\}", answer, re.DOTALL)
                if inner_match:
                    return inner_match.group(1).strip("() ")
                return answer

        # 3) 英/中 Final Answer/答案 提示样式（大小写、空格、冒号兼容）
        patterns = [
            r"(?i)final\s*answer\s*[:]\s*(.+)",
            r"(?i)answer\s*[:]\s*(.+)",
            r"(?:最终答案|答案)\s*[:]\s*(.+)",
        ]
        for p in patterns:
            m = re.search(p, output, flags=re.IGNORECASE | re.DOTALL)
            if m:
                candidate = m.group(1).strip()
                if candidate:
                    sentence = re.split(r"[\n。!?\.]", candidate, maxsplit=1)[0].strip()
                    if sentence:
                        return sentence

        # 4) 有思考标签时，取 </think> 之后首句作为兜底
        if not strict:
            if "</think>" in output:
                after_think = output.split("</think>", 1)[1].strip()
                if after_think:
                    sentence = re.split(r"[\n。!?\.]", after_think, maxsplit=1)[0].strip()
                    if sentence:
                        return sentence

            # 5) 纯文本兜底：取首个非空行/首句
            clean = (output or "").strip()
            if clean:
                sentence = re.split(r"[\n。!?\.]", clean, maxsplit=1)[0].strip()
                if sentence:
                    return sentence

        return "no answer found"

    # 终止条件，子类实现
    def is_finished(self, output:str) -> bool:
        # 基于字符串的简单完成判断
        if "<answer>" in output:
            return True

        if re.search(r"\\boxed\{.*?\}", output):
            return True

        # Final Answer/答案 样式
        if re.search(r"(?i)final\s*answer\s*[:]", output):
            return True
        if re.search(r"(?i)\banswer\s*[:]", output):
            return True
        if re.search(r"(最终答案|答案)\s*[:]", output):
            return True

        return False
