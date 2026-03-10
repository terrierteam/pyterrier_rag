import pytest

from pyterrier_rag._datasets import FlashRAGDataset


def test_flashrag_get_corpus_iter_raises_for_missing_corpus():
    dataset = FlashRAGDataset({"name": "Musique", "corpus_name": "None"})

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
