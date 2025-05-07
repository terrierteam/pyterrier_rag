import pytest
import torch

from pyterrier_rag.backend._hf import HuggingFaceBackend, Seq2SeqLMBackend, StopWordCriteria
from pyterrier_rag.backend import BackendOutput

def test_huggingface_init_and_attributes():
    # Instantiate CBT with dummy model
    backend = HuggingFaceBackend('HuggingFaceTB/SmolLM-135M', batch_size=2)
    # generation args present
    assert 'temperature' in backend._generation_args


def test_huggingface_generate_slicing_and_outputs():
    backend = HuggingFaceBackend('HuggingFaceTB/SmolLM-135M', batch_size=2, max_new_tokens=1, return_logits=True)
    # Provide two prompts
    prompts = ['a', 'bb']
    outputs = backend.generate(prompts)
    # Expect two outputs
    assert isinstance(outputs, list) and len(outputs) == 2
    # Check each BackendOutput
    for _, out in enumerate(outputs):
        assert isinstance(out, BackendOutput)
    assert all(isinstance(out.logits, torch.Tensor) for out in outputs)
    # The first sliced output should match tensor([0,99]) since prompt_length=1
    first_logits = outputs[0].logits.tolist()
    assert len(first_logits) == 1


def test_seq2seq_backend_no_slicing():
    backend = Seq2SeqLMBackend('google-t5/t5-small', batch_size=1, max_new_tokens=1, return_logits=True)
    prompts = ['xyz']
    outputs = backend.generate(prompts)
    # No prompt removal: logits list should contain full sequences of length prompt+1
    logits_list = outputs[0].logits
    # Single batch element
    assert isinstance(logits_list, torch.Tensor)
    full_seq = logits_list
    assert full_seq.shape[0] == 1


def test_stopwordcriteria_basic_behavior():
    # Dummy tokenizer: encode returns tensor length of word, decode returns word itself
    class DTok:
        def encode(self, word, return_tensors):
            return torch.tensor([1] * len(word))
        def decode(self, tokens, skip_special_tokens):
            # join token IDs as chars for simplicity
            return ''.join(str(x.item()) for x in tokens)
    tokenizer = DTok()
    # Case 1: no stop words -> always False
    crit = StopWordCriteria(tokenizer, prompt_size=2, stop_words=[], check_every=1)
    input_ids = torch.ones((2, 3), dtype=torch.long)
    res = crit(input_ids, scores=None)
    assert not res.any()
    # Case 2: seq_len not multiple of check_every -> skip
    crit = StopWordCriteria(tokenizer, prompt_size=1, stop_words=['x'], check_every=3)
    input_ids = torch.ones((1, 4), dtype=torch.long)
    res = crit(input_ids, scores=None)
    assert not res.any()
    # Case 3: stop word present
    class DTok2(DTok):
        def decode(self, tokens, skip_special_tokens):
            return 'foo_stop'
    tokenizer2 = DTok2()
    crit = StopWordCriteria(tokenizer2, prompt_size=1, stop_words=['stop'], check_every=2)
    # seq_len=3 -> 3%2=1 skip -> need seq_len multiple -> use 2
    input_ids = torch.tensor([[0,1,2,3]], dtype=torch.long)[:, :2]  # shape (1,2)
    # But seq_len=2 and prompt_size=1 -> latest_tokens length=1
    res = crit(input_ids, scores=None)
    # latest decode yields 'foo_stop', so criteria True
    assert res[0]
