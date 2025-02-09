import pytest
import pandas as pd
from unittest.mock import Mock
from pyterrier_rag.prompt import PromptTransformer


def test_initialization_requires_model_or_template():
    with pytest.raises(AssertionError, match="Either model_name_or_path or conversation_template must be provided"):
        PromptTransformer()

def test_initialization_with_model(mocker):
    mock_template = Mock()
    mocker.patch('fastchat.get_conversation_template', return_value=mock_template)
    pt = PromptTransformer(model_name_or_path='test-model')
    assert pt.model_name_or_path == 'test-model'
    assert pt.conversation_template == mock_template

def test_initialization_with_custom_template():
    mock_template = Mock()
    pt = PromptTransformer(conversation_template=mock_template)
    assert pt.conversation_template == mock_template

def test_system_message_set(mocker):
    mock_template = Mock()
    mocker.patch('fastchat.get_conversation_template', return_value=mock_template)
    system_msg = "Test system message"
    pt = PromptTransformer(model_name_or_path='test-model', system_message=system_msg)
    mock_template.set_system_message.assert_called_once_with(system_msg)

def test_instruction_as_string(mocker):
    mock_template = Mock()
    mocker.patch('fastchat.get_conversation_template', return_value=mock_template)
    instruction = "Query: {query}"
    pt = PromptTransformer(
        model_name_or_path='test-model',
        instruction=instruction,
        relevant_fields=['query']
    )
    pt.create_prompt({'query': 'test_query'})
    mock_template.append_message.assert_called_once_with('user', "Query: test_query")

def test_instruction_as_callable(mocker):
    mock_template = Mock()
    mocker.patch('fastchat.get_conversation_template', return_value=mock_template)
    def instruction_fn(query):
        return f"Search: {query}"
    pt = PromptTransformer(
        model_name_or_path='test-model',
        instruction=instruction_fn,
        relevant_fields=['query']
    )
    pt.create_prompt({'query': 'test_query'})
    mock_template.append_message.assert_called_once_with('user', "Search: test_query")

def test_output_attribute_with_api_type(mocker):
    mocker.patch('fastchat.get_conversation_template', return_value=Mock())
    pt = PromptTransformer(model_name_or_path='test-model', api_type='openai')
    assert pt.output_attribute == 'to_openai_api_messages'

def test_output_attribute_default(mocker):
    mocker.patch('fastchat.get_conversation_template', return_value=Mock())
    pt = PromptTransformer(model_name_or_path='test-model')
    assert pt.output_attribute == 'get_prompt'

def test_transform_dataframe():
    df = pd.DataFrame({
        'query': ['q1', 'q2'],
        'context': ['c1', 'c2'],
        'other': ['o1', 'o2']
    })
    pt = PromptTransformer(
        model_name_or_path='test-model',
        relevant_fields=['query', 'context'],
        output_field='prompt'
    )
    transformed_df = pt.transform(df.copy())
    for _, row in transformed_df.iterrows():
        assert row['prompt'] == {'query': row['query'], 'context': row['context']}

def test_transform_by_query(mocker):
    mock_template = Mock()
    mocker.patch('fastchat.get_conversation_template', return_value=mock_template)
    mock_prompt = Mock()
    mock_template.copy.return_value = mock_prompt
    pt = PromptTransformer(
        model_name_or_path='test-model',
        relevant_fields=['query', 'context'],
        output_field='prompt'
    )
    input_data = [{'query': 'q1', 'context': 'c1', 'docno': 'd1'}]
    output = pt.transform_by_query(input_data)
    assert output['prompt'] == mock_prompt
    assert output['query'] == 'q1'
    assert output['context'] == 'c1'
    assert output['docno'] == 'd1'
