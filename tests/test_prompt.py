import pytest
from typing import List, Optional, Dict

import pyterrier as pt
import fastchat.model as fc_model

from pyterrier_rag.prompt import PromptTransformer


class DummyTemplate:
    """Minimal conversation‐template stub for PromptTransformer tests."""
    def __init__(self):
        self.system_message: Optional[str] = None
        self.messages: List[tuple] = []

    def set_system_message(self, msg: str):
        self.system_message = msg

    def copy(self) -> "DummyTemplate":
        # Return a fresh copy so mutated state doesn’t leak
        new = DummyTemplate()
        new.system_message = self.system_message
        return new

    def append_message(self, role: str, content: str):
        self.messages.append((role, content))

    def get_prompt(self) -> str:
        # return a simple string combining all messages
        return "|".join(f"{r}:{c}" for r, c in self.messages)

    def to_openai_api_messages(self) -> List[Dict[str, str]]:
        return [{"role": r, "content": c} for r, c in self.messages]

    def to_gemini_api_messages(self) -> List[Dict[str, str]]:
        return [{"g_role": r, "g_content": c} for r, c in self.messages]

    def to_vertex_api_messages(self) -> List[Dict[str, str]]:
        return [{"v_role": r, "v_content": c} for r, c in self.messages]

    def to_reka_api_messages(self) -> List[Dict[str, str]]:
        return [{"rk": (r, c)} for r, c in self.messages]


@pytest.fixture(autouse=True)
def patch_get_conversation(monkeypatch):
    """Patch get_conversation_template to always return a fresh DummyTemplate."""
    monkeypatch.setattr(
        fc_model,
        "get_conversation_template",
        lambda model_path: DummyTemplate()
    )
    yield


def test_post_init_and_system_message():
    # system_message should be set on the template
    tf = PromptTransformer(
        instruction=lambda **f: "x",
        model_name_or_path="anything",
        system_message="SYS"
    )
    # after init, the conversation_template has the system message applied
    assert tf.conversation_template.system_message == "SYS"
    # default output attribute when api_type is None
    assert tf.output_attribute == "get_prompt"


@pytest.mark.parametrize("api,attr,method", [
    ("openai", "to_openai_api_messages", DummyTemplate.to_openai_api_messages),
    ("gemini", "to_gemini_api_messages", DummyTemplate.to_gemini_api_messages),
    ("vertex", "to_vertex_api_messages", DummyTemplate.to_vertex_api_messages),
    ("reka", "to_reka_api_messages", DummyTemplate.to_reka_api_messages),
    (None, "get_prompt", DummyTemplate.get_prompt),
])
def test_set_output_attribute(api, attr, method):
    tf = PromptTransformer(instruction=lambda **f: "i")
    tf.set_output_attribute(api)
    assert tf.output_attribute == attr

    # build a dummy prompt and call to_output
    dt = DummyTemplate()
    dt.append_message("user", "hello")
    out = tf.to_output(dt)
    # ensure dispatch is correct by comparing method result
    expected = getattr(dt, attr)()
    assert out == expected


def test_create_prompt_appends_and_returns():
    # instruction concatenates fields
    instr = lambda query, qcontext: f"{query}-{qcontext}"
    tf = PromptTransformer(instruction=instr)
    # build fields
    fields = {"query": "Q", "qcontext": "C"}
    out = tf.create_prompt(fields)
    # dummy get_prompt returns "Q-C"
    assert out == "Q-C"


def test_transform_by_query_basic():
    # instruction uses both fields
    instr = lambda query, qcontext: f"ask: {query} | {qcontext}"
    tf = PromptTransformer(instruction=instr, input_fields=["query", "qcontext"])
    inp = [{"qid": "1", "query": "foo", "qcontext": "bar"}]
    result = tf.transform_by_query(inp)
    # only one output dict
    assert isinstance(result, list) and len(result) == 1
    row = result[0]
    # output_field defaults to "prompt"
    assert "prompt" in row and row["qid"] == "1" and row["query_0"] == "foo"
    # ensure the prompt contains our instruction
    assert "ask: foo | bar" in row["prompt"]
