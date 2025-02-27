import pytest
from unittest.mock import MagicMock
import pandas as pd
from pyterrier_rag.prompt import PromptConfig, ContextConfig
from pyterrier_rag._frameworks import IRCOT

@pytest.fixture
def mock_retriever():
    mock = MagicMock()
    mock.search = MagicMock(return_value=pd.DataFrame({
        "qid": ["q1", "q1"],
        "docno": ["d1", "d2"],
        "text": ["This is document 1", "This is document 2"],
        "score": [1.0, 0.9],
    }))
    return mock

@pytest.fixture
def mock_reader():
    mock = MagicMock()
    mock.transform = MagicMock(return_value=pd.DataFrame({
        "qanswer": ["Intermediate thought"]
    }))
    mock.model_name_or_path = "mock_model"
    mock.tokenizer = MagicMock()
    mock.max_input_length = 512
    return mock

@pytest.fixture
def mock_prompt():
    mock = MagicMock()
    mock.transform = MagicMock(side_effect=lambda docs: docs)
    return mock

@pytest.fixture
def ircot_instance(mock_retriever, mock_reader, mock_prompt):
    return IRCOT(
        retriever=mock_retriever,
        reader=mock_reader,
        prompt=mock_prompt,
        max_docs=2,
        max_iterations=3,
    )

# Test Initialization
def test_initialization(mock_retriever, mock_reader, mock_prompt):
    ircot = IRCOT(
        retriever=mock_retriever,
        reader=mock_reader,
        prompt=mock_prompt,
    )

    assert ircot.max_docs == 10  # Default value
    assert ircot.max_iterations == -1  # Default value

# Test default prompt and context configurations
def test_default_configurations(mock_retriever, mock_reader):
    ircot = IRCOT(retriever=mock_retriever, reader=mock_reader)
    assert isinstance(ircot.prompt_config, PromptConfig)
    assert isinstance(ircot.context_config, ContextConfig)

# Test transform_by_query logic
def test_transform_by_query_basic(ircot_instance, mock_retriever, mock_reader):
    input_data = [{"qid": "q1", "query": "What is the capital of France?"}]

    result = ircot_instance.transform_by_query(input_data)

    assert len(result) == 1
    assert result[0]["qid"] == "q1"
    assert result[0]["query"] == "What is the capital of France?"
    assert "Intermediate thought" in result[0]["qanswer"]

    # Check retriever and reader calls
    ircot_instance.retriever.search.assert_called()
    ircot_instance.reader.transform.assert_called()

# Test exceeding max iterations
def test_transform_by_query_max_iterations(ircot_instance, mock_retriever, mock_reader):
    ircot_instance.max_iterations = 2

    input_data = pd.DataFrame.from_records([{"qid": "q1", "query": "What is the capital of France?"}])
    result = ircot_instance.transform(input_data)

    assert len(result) == 1
    assert result.iloc[0]["qid"] == "q1"
    assert "Intermediate thought" in result.iloc[0]["qanswer"]

    # Ensure iterations were limited
    assert ircot_instance.reader.transform.call_count == 2

# Test exit condition
def test_transform_by_query_exit_condition(mock_retriever, mock_reader):
    mock_reader.transform = MagicMock(
        side_effect=[
            pd.DataFrame({"qanswer": ["Intermediate thought"]}),
            pd.DataFrame({"qanswer": ["So the answer is Paris."]}),
        ]
    )

    ircot = IRCOT(retriever=mock_retriever, reader=mock_reader, max_iterations=5)

    input_data = pd.DataFrame.from_records([{"qid": "q1", "query": "What is the capital of France?"}])
    result = ircot.transform(input_data)

    assert len(result) == 1
    assert "So the answer is Paris." in result.iloc[0]["qanswer"]

    # Ensure the loop exited early
    assert mock_reader.transform.call_count == 2
