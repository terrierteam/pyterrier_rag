import pandas as pd
import pyterrier as pt
from pyterrier.datasets import Dataset

# TODO requires transformers to be installed

class RAGDataset(Dataset):
    def get_answers(self, variant=None) -> pd.DataFrame:
        pass

class FlashRAGDataset(RAGDataset):
    def __init__(self, flashsplits):
        self.splits = flashsplits
        # TODO: we should cache the df?

    def get_topics(self, variant=None) -> pd.DataFrame:
        df = pd.read_json("hf://datasets/RUC-NLPIR/FlashRAG_datasets/" + self.splits[variant], lines=True)
        return df[['id', 'question']].rename(columns={'id' : 'qid', 'question' : 'query'})

    def get_answers(self, variant=None) -> pd.DataFrame:
        df = pd.read_json("hf://datasets/RUC-NLPIR/FlashRAG_datasets/" + self.splits[variant], lines=True)
        return df[['id', 'golden_answers']].rename(columns={'id' : 'qid', 'golden_answers' : 'answers'}) # TODO how to deal with multiple answers

# TODO perhaps this should be done with entrypoints?
pt.datasets.DATASET_MAP['rag:nq'] = FlashRAGDataset({'train': 'nq/train.jsonl', 'dev': 'nq/dev.jsonl', 'test': 'nq/test.jsonl'})
pt.datasets.DATASET_MAP['rag:hotpotqa'] = FlashRAGDataset({'train': 'hotpotqa/train.jsonl', 'dev': 'hotpotqa/dev.jsonl'})