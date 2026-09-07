from typing import Dict, Iterable, Optional
from warnings import warn

import pandas as pd
import pyterrier as pt
from pyterrier.datasets import Dataset, RemoteDataset

# TODO requires transformers to be installed

class RAGDataset(Dataset):
    def get_answers(self, variant: Optional[str] = None) -> pd.DataFrame:
        pass

class RemoteRAGDataset(RemoteDataset, RAGDataset):
    def get_answers(self, variant : Optional[str] = None):
        filename, type = self._get_one_file("answers", variant)
        if type == "direct":
            return filename
        return pt.io.read_qrels(filename)

class FlashRAGDataset(RAGDataset):
    def __init__(self, flashsplits : Dict[str,str]):
        self.splits = flashsplits
        # TODO: we should cache the df?
    
    def get_corpus_iter(self) -> pt.model.IterDict:
        if self.splits['corpus_name'] is None:
            name = self.splits['name']
            raise NotImplementedError(f'{self!r} for {name} does not support get_corpus_iter')
        return pt.get_dataset(self.splits['corpus_name']).get_corpus_iter()

    def get_topics(self, variant : Optional[str] = None) -> pd.DataFrame:
        if variant is None:
            raise ValueError("Variant must be specified for %s: %s" % (self.name, str(self.splits.keys())))
        df = pd.read_json("hf://datasets/RUC-NLPIR/FlashRAG_datasets/" + self.splits[variant], lines=True)
        return df[['id', 'question']].rename(columns={'id' : 'qid', 'question' : 'query'})

    def get_answers(self, variant : Optional[str] = None) -> pd.DataFrame:
        if variant is None:
            raise ValueError("Variant must be specified for %s: %s" % (self.name, str(self.splits.keys())))
        df = pd.read_json("hf://datasets/RUC-NLPIR/FlashRAG_datasets/" + self.splits[variant], lines=True)
        return df[['id', 'golden_answers']] \
            .rename(columns={'id' : 'qid', 'golden_answers' : 'gold_answer'}) \
            .explode('gold_answer') # explode deals with multiple answers

class TopiOCQADataset(RAGDataset):
    """TopiOCQA conversational question answering dataset."""

    SPLITS = {
        'train': 'topiocqa_train.jsonl',
        'dev': 'topiocqa_valid.jsonl',
        'valid': 'topiocqa_valid.jsonl',
        'validation': 'topiocqa_valid.jsonl',
    }

    BASE_PATH = "hf://datasets/McGill-NLP/TopiOCQA/data/"

    def _read_split(self, variant : Optional[str] = None) -> pd.DataFrame:
        if variant is None:
            raise ValueError("Variant must be specified for TopiOCQA: train or dev")
        if variant not in self.SPLITS:
            raise ValueError(
                "Unknown TopiOCQA variant %r. Expected one of %s"
                % (variant, str(list(self.SPLITS.keys())))
            )
        return pd.read_json(
            self.BASE_PATH + self.SPLITS[variant],
            lines=True
        )

    @staticmethod
    def _add_qid(df : pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['qid'] = (
            df['Conversation_no'].astype(str)
            + '_'
            + df['Turn_no'].astype(str)
        )
        return df

    def get_topics(self, variant : Optional[str] = None) -> pd.DataFrame:
        df = self._add_qid(self._read_split(variant))
        return df[
            [
                'qid',
                'Question',
                'Context',
                'Conversation_no',
                'Turn_no'
            ]
        ].rename(
            columns={
                'Question': 'query',
                'Context': 'history',
                'Conversation_no': 'conversation_id',
                'Turn_no': 'turn_id'
            }
        )

    def get_answers(self, variant : Optional[str] = None) -> pd.DataFrame:
        df = self._add_qid(self._read_split(variant))
        rows = []

        for row in df.itertuples():
            answers = [row.Answer]
            additional_answers = row.Additional_answers

            if isinstance(additional_answers, dict):
                if 'Answer' in additional_answers:
                    extra = additional_answers['Answer']
                    if isinstance(extra, list):
                        answers.extend(extra)
                    elif extra is not None:
                        answers.append(extra)
                else:
                    for extra in additional_answers.values():
                        if isinstance(extra, list):
                            answers.extend(extra)
                        elif extra is not None:
                            answers.append(extra)

            elif isinstance(additional_answers, list):
               for extra in additional_answers:
                  if isinstance(extra, dict):
                     answer = extra.get('Answer')
                     if answer is not None:
                         answers.append(answer)
                  elif extra is not None:
                       answers.append(extra)

            elif additional_answers is not None:
                answers.append(additional_answers)

            for answer in answers:
                rows.append({
                    'qid': row.qid,
                    'gold_answer': answer
                })

        return pd.DataFrame(
            rows,
            columns=['qid', 'gold_answer']
        )

    def get_qrels(self, variant : Optional[str] = None) -> pd.DataFrame:
        df = self._add_qid(self._read_split(variant))
        rows = []

        for row in df.itertuples():
            gold_passage = row.Gold_passage

            if isinstance(gold_passage, dict):
                docno = gold_passage.get('id')
            else:
                docno = None

            if docno is not None:
                rows.append({
                    'qid': row.qid,
                    'docno': str(docno),
                    'label': 1
                })

        return pd.DataFrame(
            rows,
            columns=['qid', 'docno', 'label']
        )

DATASET_MAP = {}

DATASET_MAP['topiocqa'] = TopiOCQADataset()

DATASET_MAP['nq'] = FlashRAGDataset(
    {'name': 'NQ', 'corpus_name': 'irds:beir/nq', 'train': 'nq/train.jsonl', 'dev': 'nq/dev.jsonl', 'test': 'nq/test.jsonl'})
DATASET_MAP['hotpotqa'] = FlashRAGDataset(
    {'name': 'HotPotQA', 'corpus_name': 'irds:beir/hotpotqa', 'train': 'hotpotqa/train.jsonl', 'dev': 'hotpotqa/dev.jsonl'})
DATASET_MAP['triviaqa'] = FlashRAGDataset(
    {'name': 'TriviaQA', 'corpus_name': 'irds:dpr-w100/trivia-qa/dev', 'train': 'triviaqa/train.jsonl', 'dev': 'triviaqa/dev.jsonl', 'test': 'triviaqa/test.jsonl'})
DATASET_MAP['musique'] = FlashRAGDataset(
    {'name': 'Musique', 'corpus_name': None, 'train': 'musique/train.jsonl', 'dev': 'musique/dev.jsonl'})
pt.datasets.DATASET_MAP['rag:web_questions'] = FlashRAGDataset(
    {'name': 'Web Questions', 'corpus_name': None, 'train': 'web_questions/train.jsonl', 'test': 'web_questions/test.jsonl'})
pt.datasets.DATASET_MAP['rag:wow'] = FlashRAGDataset(
    {'name': 'WoW', 'corpus_name': None, 'train': 'wow/train.jsonl', 'dev': 'wow/dev.jsonl'})
pt.datasets.DATASET_MAP['rag:popqa'] = FlashRAGDataset(
    {'name': 'PopQA', 'corpus_name': None, 'test': 'popqa/test.jsonl'})


def _hotspot_files(dataset: Dataset, components: str, variant: str, **kwargs):
    tar_name = 'enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2'

    # This is equivalent code to extract
    # localtarfile, _ = dataset._get_one_file("tars", tar_name)
    # import tarfile
    # tarf = tarfile.open(localtarfile, 'r:bz2')
    # all_members = tarf.getmembers()
    # all_files = [(info.name.replace("/", "_"), tar_name + '#' + info.name)
    #   for info in all_members if '.bz2' in info.name and info.isfile()]

    import os
    file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "etc",
        "rag_hotpotqa_wiki.files.txt")
    with open(file, "rt") as f:
        all_files = [ (name.strip().replace("/", "_"), tar_name + '#' + name.strip()) for name in f ]
    return all_files

def _hotpotread_iterator(dataset):

    del_keys = ['charoffset_with_links', 'text_with_links', 'charoffset']
    import bz2
    import json
    for filename in dataset.get_corpus():

        with bz2.open(filename, 'rt') as f:
            for lineno, line in enumerate(f):
                try:
                    line_dict = json.loads(line)
                    if not isinstance(line_dict, dict):
                        raise json.decoder.JSONDecodeError("Not a dict", line, lineno)
                    line_dict["docno"] = line_dict.pop("id")
                    line_dict['text'] = ' '.join(line_dict['text'])
                    for k in del_keys:
                        del line_dict[k]
                    yield line_dict
                except json.decoder.JSONDecodeError as jse:
                    if lineno > 10:
                        warn("Ignoring JSON decoding error on line number %d, line %sm error %s" % (lineno, line, str(jse)))
                    else:
                        raise jse

HOTPOTQA_WIKI = {
    "tars" : {
        'enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2' : ( 'enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2', 'https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2' )
    },
    "corpus" :_hotspot_files,
    "corpus_iter" : _hotpotread_iterator
}

DATASET_MAP['hotpotqa_wiki'] = RemoteDataset('rag:hotpotqa_wiki', HOTPOTQA_WIKI)

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

DATASET_MAP['nq_wiki'] = RemoteDataset('rag:nq_wiki', FLASHRAG_WIKI)

def _2WikiMultihopQA_topics(self, component, variant):
    assert component in ['topics', 'answers']
    json_file, _ = self._get_one_file('raw_files', variant)
    all_data = pd.read_json(json_file)
    if component == 'answers':
        answers = all_data.rename(columns={'_id': 'qid', 'answer' : 'gold_answer'})[['qid', 'type', 'gold_answer']]
        return answers, "direct"
    rtr = []
    for id, idgroup in pt.tqdm(all_data.explode('context').groupby('_id'), desc="Reading 2WikiMultihopQA %s.json" % variant):
        for docpos, doc in enumerate(idgroup.itertuples()):
            rtr.append({
                'qid' : id,
                'query' : doc.question,
                'docno' : "%s_%02d" % (id, docpos),
                'title' : doc.context[0],
                'text' : " ".join(doc.context[1]) # join the sentences into a single passage
                })

    return pd.DataFrame(rtr), "direct"

DROPBOX_2WikiMultihopQA = {
    "tars" : {
        '2WikiMultihopQA.zip' : ('2WikiMultihopQA.zip', "https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?dl=1")
    },
    "raw_files" : {
        'test' : ('test.json', '2WikiMultihopQA.zip#test.json'),
        'train' : ('train.json', '2WikiMultihopQA.zip#train.json'),
        'dev' : ('dev.json', '2WikiMultihopQA.zip#dev.json')
    },
    "topics" : {
        'train' : _2WikiMultihopQA_topics,
        'dev' : _2WikiMultihopQA_topics,
        'test' : _2WikiMultihopQA_topics,
    },
    "answers" : {
        'train' : _2WikiMultihopQA_topics,
        'dev' : _2WikiMultihopQA_topics,
        # no answers in the test set
    }
}
DATASET_MAP['2wikimultihopqa'] = RemoteRAGDataset('rag:2wikimultihopqa', DROPBOX_2WikiMultihopQA)


if hasattr(pt.datasets, 'DatasetProvider'):
    # PyTerrier version that supports DatasetProviders
    class RagDatasetProvider(pt.datasets.DatasetProvider):
        def get_dataset(self, name: str) -> pt.datasets.Dataset:
            return DATASET_MAP[name]

        def list_datasets(self) -> Iterable[str]:
            return list(DATASET_MAP.keys())
else:
    # Fallback: manually change PyTerrier core's DATASET_MAP. (This requires that this module be loaded before
    # these datasets are available)
    pt.datasets.DATASET_MAP.update({f'rag:{k}': v for k, v in DATASET_MAP.items()})
