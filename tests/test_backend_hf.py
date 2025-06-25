import unittest
import pytest
import torch

from pyterrier_rag.backend._hf import HuggingFaceBackend, Seq2SeqLMBackend, StopWordCriteria
from pyterrier_rag.backend import BackendOutput, Backend
from . import test_backend


class TestHuggingFaceBackend(test_backend.BaseTestBackend, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = Backend.from_dsn('huggingface:HuggingFaceTB/SmolLM-135M')

    def test_huggingface_init_and_attributes(self):
        # generation args present
        assert 'temperature' in self.backend._generation_args

    # TODO: once return_logprobs support added to HuggingFaceBackend
    # def test_huggingface_generate_slicing_and_outputs(self):
    #     # Provide two prompts
    #     prompts = ['a', 'bb']
    #     outputs = self.backend.generate(prompts, max_new_tokens=1, return_logprobs=True)
    #     # Expect two outputs
    #     assert isinstance(outputs, list) and len(outputs) == 2
    #     # Check each BackendOutput
    #     for _, out in enumerate(outputs):
    #         assert isinstance(out, BackendOutput)
    #     assert all(isinstance(out.logprobs, list) for out in outputs)
    #     # The first sliced output should match tensor([0,99]) since prompt_length=1
    #     first_logprobs = outputs[0].logprobs.tolist()
    #     assert len(first_logprobs) == 1


class TestHuggingFaceBackendSeq2Seq(test_backend.BaseTestBackend, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = Seq2SeqLMBackend('google-t5/t5-small')

    # TODO: once return_logprobs support added to Seq2SeqLMBackend
    # def test_seq2seq_backend_no_slicing(self):
    #     prompts = ['xyz']
    #     outputs = self.backend.generate(prompts, max_new_tokens=1, return_logprobs=True)
    #     # No prompt removal: logprobs list should contain full sequences of length prompt+1
    #     logprobs_list = outputs[0].logprobs
    #     # Single batch element
    #     assert isinstance(logprobs_list, list)
    #     full_seq = logprobs_list
    #     assert full_seq.shape[0] == 1





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
