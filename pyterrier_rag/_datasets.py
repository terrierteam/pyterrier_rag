import pandas as pd
import pyterrier as pt
from pyterrier.datasets import Dataset, RemoteDataset
from typing import Optional, Dict
from warnings import warn

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
pt.datasets.DATASET_MAP['rag:2wikimultihopqa'] = FlashRAGDataset(
    {'train': '2wikimultihopqa/train.jsonl', 'dev': '2wikimultihopqa/dev.jsonl'})
pt.datasets.DATASET_MAP['rag:triviaqa'] = FlashRAGDataset(
    {'train': 'triviaqa/train.jsonl', 'dev': 'triviaqa/dev.jsonl', 'test': 'triviaqa/test.jsonl'})

def _hotspot_files(dataset, components, variant, **kwargs):
    TAR_NAME = 'enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2'
    
    # This is equivalent code to extract 
    # localtarfile, _ = dataset._get_one_file("tars", TAR_NAME)
    # import tarfile
    # tarf = tarfile.open(localtarfile, 'r:bz2')
    # all_members = tarf.getmembers()
    # # we replace / in the local name, as pyterrier doesnt support /
    # all_files = [(info.name.replace("/", "_"), TAR_NAME + '#' + info.name) for info in all_members if '.bz2' in info.name and info.isfile()]
    
    import os
    file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "etc",
        "rag:hotpotqa_wiki.files.txt")
    with open(file, "rt") as f:
        all_files = [ (name.strip().replace("/", "_"), TAR_NAME + '#' + name.strip()) for name in f ]
    return all_files

def _hotpotread_iterator(dataset):

    DEL_KEYS = ['charoffset_with_links', 'text_with_links', 'charoffset']
    import bz2, json
    for filename in dataset.get_corpus():

        with bz2.open(filename, 'rt') as f:
            for lineno, line in enumerate(f):
                try:
                    line_dict = json.loads(line)
                    if not isinstance(line_dict, dict):
                        raise json.decoder.JSONDecodeError("Not a dict", line, lineno)
                    line_dict["docno"] = line_dict.pop("id")
                    line_dict['text'] = ' '.join(line_dict['text'])
                    for k in DEL_KEYS:
                        del line_dict[k]
                    yield line_dict
                except json.decoder.JSONDecodeError as jse:
                    if lineno > 10:
                        warn("Ignoring JSON decoding error on line number %d, line %sm error %s" % (lineno, line, str(jse)))
                    else: 
                        raise jse

HOTPOTQA_WIKI = {
    "tars" : {
        'enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2' : ( 'enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2', 'http://www.dcs.gla.ac.uk/~craigm/enwiki-20171001-pages-meta-current-withlinks-abstracts.SMALL.tar.bz2' )
        # 'https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2'
    },
    "corpus" :_hotspot_files,
    "corpus_iter" : _hotpotread_iterator
}

pt.datasets.DATASET_MAP['rag:hotpotqa_wiki'] = RemoteDataset('rag:hotpotqa_wiki', HOTPOTQA_WIKI)

def _nq_read_iterator(dataset):
    import json
    for filename in dataset.get_corpus():
        with pt.io.autoopen(filename, "r", encoding="utf-8", errors='replace') as f: 
            # error='replace' avoids a UTF encoding error
            for lineno, line in enumerate(f):
                try:
                    line_dict = json.loads(line)
                    if not isinstance(line_dict, dict):
                        raise json.decoder.JSONDecodeError("Not a dict", line, lineno)
                    line_dict["docno"] = line_dict.pop("id")
                    # flashrag has {title}\n{text}
                    # we split this out, but keep contents too for anyone that wants it 
                    title, text = line_dict["contents"].split("\n", 1) 
                    line_dict['title'] = title
                    line_dict['text'] = text
                    yield line_dict
                except json.decoder.JSONDecodeError as jse:
                    if lineno > 10:
                        warn("Ignoring JSON decoding error in file %s on line number %d, line %s error %s" % (filename, lineno, line, str(jse)))
                    else: 
                        raise RuntimeError("Early JSON decoding error in file %s on line number %d, line %s" % (filename, lineno, line)) from jse

FLASHRAG_WIKI = {
    "tars" : {
        'wiki18_100w.zip' : ('wiki18_100w.zip', 'https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/retrieval-corpus/wiki18_100w.zip')
    },
    "corpus" : [("wiki18_100w.jsonl", "wiki18_100w.zip#wiki18_100w.jsonl")],
    "corpus_iter" : _nq_read_iterator
}

pt.datasets.DATASET_MAP['rag:nq_wiki'] = RemoteDataset('rag:nq_wiki', FLASHRAG_WIKI)
