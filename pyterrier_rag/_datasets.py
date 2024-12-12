import pandas as pd
import pyterrier as pt
from typing import Optional, Dict
from pyterrier.datasets import Dataset

# TODO requires transformers to be installed

class RAGDataset(Dataset):
    def get_answers(self, variant: Optional[str] = None) -> pd.DataFrame:
        pass

class FlashRAGDataset(RAGDataset):
    def __init__(self, flashsplits : Dict[str,str]):
        self.splits = flashsplits
        # TODO: we should cache the df?

    def get_topics(self, variant : Optional[str] = None) -> pd.DataFrame:
        df = pd.read_json("hf://datasets/RUC-NLPIR/FlashRAG_datasets/" + self.splits[variant], lines=True)
        return df[['id', 'question']].rename(columns={'id' : 'qid', 'question' : 'query'})

    def get_answers(self, variant : Optional[str] = None) -> pd.DataFrame:
        df = pd.read_json("hf://datasets/RUC-NLPIR/FlashRAG_datasets/" + self.splits[variant], lines=True)
        return df[['id', 'golden_answers']] \
            .rename(columns={'id' : 'qid', 'golden_answers' : 'gold_answer'}) \
            .explode('gold_answer') # explode deals with multiple answers

# TODO perhaps this should be done with entrypoints?
pt.datasets.DATASET_MAP['rag:nq'] = FlashRAGDataset(
    {'train': 'nq/train.jsonl', 'dev': 'nq/dev.jsonl', 'test': 'nq/test.jsonl'})
pt.datasets.DATASET_MAP['rag:hotpotqa'] = FlashRAGDataset(
    {'train': 'hotpotqa/train.jsonl', 'dev': 'hotpotqa/dev.jsonl'})