from unittest.mock import Mock

import pandas as pd
import pytest

from pyterrier_rag._datasets import DATASET_MAP, FlashRAGDataset


def test_flashrag_get_corpus_iter_raises_for_missing_corpus():
    dataset = FlashRAGDataset({"name": "Musique", "corpus_name": None})

    with pytest.raises(NotImplementedError, match="Musique does not support get_corpus_iter"):
        dataset.get_corpus_iter()


def test_flashrag_get_corpus_iter_delegates_to_pyterrier_dataset(monkeypatch):
    expected_iter = iter([{"docno": "d1", "text": "t1"}])

    class StubCorpusDataset:
        def get_corpus_iter(self):
            return expected_iter

    called = {}

    def fake_get_dataset(name):
        called["name"] = name
        return StubCorpusDataset()

    monkeypatch.setattr("pyterrier_rag._datasets.pt.get_dataset", fake_get_dataset)

    dataset = FlashRAGDataset({"name": "NQ", "corpus_name": "irds:beir/nq"})
    result_iter = dataset.get_corpus_iter()

    assert called["name"] == "irds:beir/nq"
    assert result_iter is expected_iter


@pytest.mark.parametrize("method", ["get_topics", "get_answers"])
def test_flashrag_missing_variant_lists_available_splits(method):
    with pytest.raises(ValueError, match=r"^Variant must be specified for NQ: train, dev, test$"):
        getattr(DATASET_MAP['nq'], method)()


def test_flashrag_explicit_split_loads_topics_and_answers(monkeypatch):
    read_json = Mock(return_value=pd.DataFrame({
        'id': ['1'], 'question': ['Who?'], 'golden_answers': [['Ada', 'A. Lovelace']],
    }))
    monkeypatch.setattr('pyterrier_rag._datasets.pd.read_json', read_json)
    dataset = DATASET_MAP['nq']

    assert dataset.get_topics('dev').to_dict('records') == [{'qid': '1', 'query': 'Who?'}]
    assert dataset.get_answers('dev').to_dict('records') == [
        {'qid': '1', 'gold_answer': 'Ada'},
        {'qid': '1', 'gold_answer': 'A. Lovelace'},
    ]
    assert read_json.call_count == 2
    for call in read_json.call_args_list:
        assert call.args == ('hf://datasets/RUC-NLPIR/FlashRAG_datasets/nq/dev.jsonl',)
        assert call.kwargs == {'lines': True}
