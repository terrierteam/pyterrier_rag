import pytest
import pandas as pd
from unittest.mock import Mock, patch
from pyterrier_rag.prompt import PromptTransformer, PromptConfig, ContextConfig

@pytest.fixture
def mock_prompt_config():
    return PromptConfig(
        instruction="Test instruction",
        model_name_or_path="mock_model",
        system_message="Test system message",
        input_fields=["field1", "field2"],
        output_field="output",
        api_type="openai",
        conversation_template=None,
    )

@pytest.fixture
def mock_context_config():
    return ContextConfig(aggregate_func=lambda x: {"context": "mock_context"})

@pytest.fixture
def mock_conversation_template():
    class MockConversationTemplate:
        def __init__(self):
            self.messages = []
        
        def copy(self):
            return MockConversationTemplate()
        
        def set_system_message(self, message):
            pass
        
        def append_message(self, role, message):
            self.messages.append({"role": role, "message": message})
        
        def get_prompt(self):
            return "mock_prompt"
        
        def to_openai_api_messages(self):
            return [{"role": "user", "content": "mock_prompt"}]
    
    return MockConversationTemplate()

@patch("fastchat.model.get_conversation_template")
def test_initialization(mock_get_template, mock_prompt_config):
    mock_get_template.return_value = None
    transformer = PromptTransformer(config=mock_prompt_config)
    assert transformer.config == mock_prompt_config
    assert transformer.output_field == "output"
    assert transformer.api_type == "openai"
    assert not transformer.use_context

@patch("fastchat.model.get_conversation_template")
def test_context_config_initialization(mock_get_template, mock_prompt_config, mock_context_config):
    mock_get_template.return_value = None
    transformer = PromptTransformer(config=mock_prompt_config, context_config=mock_context_config)
    assert transformer.use_context
    assert transformer.context_config == mock_context_config

@patch("fastchat.model.get_conversation_template")
@patch("pyterrier_rag.prompt._context_aggregation.ContextAggregationTransformer")
def test_context_aggregation_initialization(mock_context_agg, mock_get_template, mock_prompt_config, mock_context_config):
    mock_get_template.return_value = None
    mock_context_agg.return_value = Mock()
    transformer = PromptTransformer(config=mock_prompt_config, context_config=mock_context_config)
    assert transformer.context_aggregation is not None

@patch("fastchat.model.get_conversation_template")
def test_create_prompt(mock_get_template, mock_prompt_config, mock_conversation_template):
    mock_get_template.return_value = mock_conversation_template
    transformer = PromptTransformer(config=mock_prompt_config)
    fields = {"field1": "value1", "field2": "value2"}
    prompt = transformer.create_prompt(fields)
    assert isinstance(prompt, list)
    assert prompt[0]["role"] == "system"
    assert prompt[1]["role"] == "user"
    assert "Test instruction" in prompt[1]["content"]

@patch("fastchat.model.get_conversation_template")
def test_transform_by_query(mock_get_template, mock_prompt_config, mock_conversation_template):
    mock_get_template.return_value = mock_conversation_template
    transformer = PromptTransformer(config=mock_prompt_config)
    inp = pd.DataFrame.from_records([{"qid": "1", "query": "test query", "field1": "value1", "field2": "value2"}])
    result = transformer.transform(inp)
    assert len(result) == 1
    assert "output" in result.iloc[0]
    assert result.iloc[0]["qid"] == "1"
    assert result.iloc[0]["query_0"] == "test query"

@patch("fastchat.model.get_conversation_template")
def test_transform_iter(mock_get_template, mock_prompt_config, mock_conversation_template):
    mock_get_template.return_value = mock_conversation_template
    transformer = PromptTransformer(config=mock_prompt_config)
    inp = [{"qid": "1", "query": "test query", "field1": "value1", "field2": "value2"}]
    result = list(transformer.transform_iter(inp))
    assert len(result) == 1
    assert "output" in result[0]
    assert result[0]["qid"] == "1"
    assert result[0]["query_0"] == "test query"
