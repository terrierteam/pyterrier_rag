import pandas as pd
import pytest

from pyterrier_rag._datasets import DATASET_MAP, FlashRAGDataset, TopiOCQADataset


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


@pytest.fixture
def topiocqa_sample():
    return pd.DataFrame([
        {
            "Conversation_no": 100,
            "Turn_no": 1,
            "Question": "who won the bharat ratna award in 2018",
            "Answer": "UNANSWERABLE",
            "Context": [],
            "Additional_answers": [
               {
                   "Answer": "UNANSWERABLE",
                },
               {
                    "Answer": "UNANSWERABLE",
                },
               {
                     "Answer": "UNANSWERABLE",
                },
            ],
            "Gold_passage": {
                "id": "wiki:100",
                "title": "Bharat Ratna",
                "text": "Example passage one.",
            },
        },
        {
            "Conversation_no": 100,
            "Turn_no": 2,
            "Question": "when was this award initially given?",
            "Answer": "2 January 1954",
            "Context": [
                "who won the bharat ratna award in 2018",
                "UNANSWERABLE",
            ],
            "Additional_answers": [
               {
                   "Answer": "2 January 1954",
                    "Topic": "Bharat Ratna",
                },
               {
                    "Answer": "2 January 1954",
                    "Topic": "Bharat Ratna",
             },
{
        "Answer": "1954",
        "Topic": "Bharat Ratna",
    },
],
            "Gold_passage": {
                "id": "wiki:11651453",
                "title": "Bharat Ratna",
                "text": "Example passage two.",
            },
        },
    ])


def test_topiocqa_is_registered():
    assert "topiocqa" in DATASET_MAP
    assert isinstance(DATASET_MAP["topiocqa"], TopiOCQADataset)


def test_topiocqa_get_topics(monkeypatch, topiocqa_sample):
    dataset = TopiOCQADataset()

    monkeypatch.setattr(
        dataset,
        "_read_split",
        lambda variant: topiocqa_sample,
    )

    topics = dataset.get_topics("dev")

    assert list(topics.columns) == [
        "qid",
        "query",
        "history",
        "conversation_id",
        "turn_id",
    ]

    assert topics["qid"].tolist() == [
        "100_1",
        "100_2",
    ]

    row = topics[
        topics["qid"] == "100_2"
    ].iloc[0]

    assert row["query"] == "when was this award initially given?"

    assert row["history"] == [
        "who won the bharat ratna award in 2018",
        "UNANSWERABLE",
    ]

    assert row["conversation_id"] == 100
    assert row["turn_id"] == 2


def test_topiocqa_get_answers(monkeypatch, topiocqa_sample):
    dataset = TopiOCQADataset()

    monkeypatch.setattr(
        dataset,
        "_read_split",
        lambda variant: topiocqa_sample,
    )

    answers = dataset.get_answers("dev")

    assert list(answers.columns) == [
        "qid",
        "gold_answer",
    ]

    answers_100_2 = answers[
        answers["qid"] == "100_2"
    ]["gold_answer"].tolist()

    assert answers_100_2 == [
        "2 January 1954",
        "2 January 1954",
        "2 January 1954",
        "1954",
    ]


def test_topiocqa_get_qrels(monkeypatch, topiocqa_sample):
    dataset = TopiOCQADataset()

    monkeypatch.setattr(
        dataset,
        "_read_split",
        lambda variant: topiocqa_sample,
    )

    qrels = dataset.get_qrels("dev")

    assert list(qrels.columns) == [
        "qid",
        "docno",
        "label",
    ]

    assert qrels.to_dict("records") == [
        {
            "qid": "100_1",
            "docno": "wiki:100",
            "label": 1,
        },
        {
            "qid": "100_2",
            "docno": "wiki:11651453",
            "label": 1,
        },
    ]


def test_topiocqa_requires_variant():
    dataset = TopiOCQADataset()

    with pytest.raises(
        ValueError,
        match="Variant must be specified for TopiOCQA",
    ):
        dataset.get_topics()


def test_topiocqa_rejects_unknown_variant():
    dataset = TopiOCQADataset()

    with pytest.raises(
        ValueError,
        match="Unknown TopiOCQA variant",
    ):
        dataset.get_topics("test")