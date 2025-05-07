import inspect
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import pyterrier as pt
import pyterrier_alpha as pta
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    BartModel,
    GenerationConfig,
    T5ForConditionalGeneration,
)
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput, Seq2SeqModelOutput
from transformers.models.bart.configuration_bart import BartConfig
from transformers.models.t5.configuration_t5 import T5Config


@dataclass
class FiDEncoderOuput(BaseModelOutput):
    last_hidden_state: torch.FloatTensor = None
    attention_mask: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class T5FiDReader(T5ForConditionalGeneration):

    def __init__(self, config: T5Config, **kwargs):
        super().__init__(config)

    def get_encoder_output(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        **kwargs
    ) -> dict:

        need_flatten = True if len(input_ids.shape) > 2 else False
        if need_flatten:
            batch_size, num_passages, seq_length = input_ids.shape
            input_ids = input_ids.reshape(-1, seq_length) # batch_size x num_passages, seq_length
            attention_mask = attention_mask.reshape(-1, seq_length)

        encoder_outputs = self.encoder(
            input_ids = input_ids,
            attention_mask = attention_mask,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = True,
        )

        hidden_states = encoder_outputs[0] # batch_size x num_passages, seq_length, hidden_size
        hidden_size = hidden_states.shape[-1]
        if need_flatten:
            hidden_states = hidden_states.reshape(batch_size, num_passages*seq_length, hidden_size)
            attention_mask = attention_mask.reshape(batch_size, num_passages*seq_length)

        outputs = FiDEncoderOuput(
            last_hidden_state=hidden_states,
            attention_mask=attention_mask,
            hidden_states=encoder_outputs.hidden_states, 
            attentions=encoder_outputs.attentions
        )
        return outputs

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Dict[str, torch.Tensor]] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = False,
        **kwargs
    ):

        if encoder_outputs is None:
            encoder_outputs = self.get_encoder_output(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                **kwargs
            )

        encoder_hidden_states = encoder_outputs.last_hidden_state
        encoder_attention_mask = encoder_outputs.attention_mask

        # for decoding
        if labels is not None:
            decoder_input_ids = self._shift_right(labels)

        decoder_output = self.decoder(
            input_ids = decoder_input_ids,
            attention_mask = decoder_attention_mask,
            encoder_hidden_states = encoder_hidden_states,
            encoder_attention_mask = encoder_attention_mask,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = False,
        )

        sequence_output = decoder_output[0]
        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim**-0.5)
        if sequence_output.dtype != self.lm_head.weight.dtype:
            sequence_output = sequence_output.to(self.lm_head.weight.dtype)
        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.reshape(-1, lm_logits.shape[-1]), labels.reshape(-1))

        if not return_dict:
            output = (lm_logits, encoder_hidden_states)
            output = ((loss, ) + output) if loss is not None else output
            return output

        return Seq2SeqLMOutput(loss=loss, logits=lm_logits)

    def generate(self, **kwargs):

        input_ids = kwargs.pop("input_ids")
        attention_mask = kwargs.pop("attention_mask")
        encoder_outputs = self.get_encoder_output(
            input_ids=input_ids, 
            attention_mask=attention_mask,
        )
        kwargs["encoder_outputs"] = encoder_outputs
        return super().generate(**kwargs)

    def _prepare_encoder_decoder_kwargs_for_generation(self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name, *args, **kwargs):

        # 1. get encoder
        # encoder = self.get_encoder()

        # 2. Prepare encoder args and encoder kwargs from model kwargs.
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        # encoder_signature = set(inspect.signature(encoder.forward).parameters)
        encoder_signature = set(inspect.signature(self.get_encoder_output).parameters)

        encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        if not encoder_accepts_wildcard:
            encoder_kwargs = {
                argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature
            }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        # model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)
        model_kwargs["encoder_outputs"] = self.get_encoder_output(**encoder_kwargs)

        # dict_keys(['attention_mask', 'ent_indices', 'ent_mask', 'output_attentions', 'output_hidden_states', 'use_cache', 'encoder_outputs'])
        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ):

        return {
            "encoder_outputs": encoder_outputs,
            "decoder_input_ids": input_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class FiDBartTModel(BartModel):

    def get_encoder_output(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ):
        need_flatten = True if len(input_ids.shape) > 2 else False
        if need_flatten:
            batch_size, num_passages, seq_length = input_ids.shape
            input_ids = input_ids.reshape(-1, seq_length)
            attention_mask = attention_mask.reshape(-1, seq_length)

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if return_dict:
            hidden_states = encoder_outputs.last_hidden_state
            all_hidden_states = encoder_outputs.hidden_states
            all_attentions = encoder_outputs.attentions
        else:
            hidden_states = encoder_outputs[0]
            all_hidden_states = encoder_outputs[1] if len(encoder_outputs) > 1 else None
            all_attentions = encoder_outputs[2] if len(encoder_outputs) > 2 else None

        hidden_size = hidden_states.shape[-1]
        if need_flatten:
            hidden_states = hidden_states.reshape(batch_size, num_passages*seq_length, hidden_size)
            attention_mask = attention_mask.reshape(batch_size, num_passages*seq_length)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

        return FiDEncoderOuput(
            last_hidden_state=hidden_states, 
            attention_mask=attention_mask,
            hidden_states=all_hidden_states, 
            attentions=all_attentions
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # copy from https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/bart/modeling_bart.py#L838
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.get_encoder_output(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        # elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
        #     encoder_outputs = BaseModelOutput(
        #         last_hidden_state=encoder_outputs[0],
        #         hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
        #         attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
        #     )
        encoder_hidden_states = encoder_outputs.last_hidden_state
        encoder_attention_mask = encoder_outputs.attention_mask

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states, # encoder_outputs[0],
            encoder_attention_mask=encoder_attention_mask, # attention_mask.reshape(attention_mask.shape[0], -1),
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class BARTFiDReader(BartForConditionalGeneration):

    def __init__(self, config: BartConfig, **kwargs):

        super().__init__(config)
        self.model = FiDBartTModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder_output(self, *args, **kwargs):
        return self.model.get_encoder_output(*args, **kwargs)
    
    def generate(self, **kwargs):

        input_ids = kwargs.pop("input_ids")
        attention_mask = kwargs.pop("attention_mask")
        encoder_outputs = self.get_encoder_output(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            return_dict=True,
        )
        kwargs["encoder_outputs"] = encoder_outputs
        return super().generate(**kwargs)

    def _prepare_encoder_decoder_kwargs_for_generation(self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name, *args, **kwargs):

        # 1. get encoder
        # encoder = self.get_encoder()
        encoder = self.model.get_encoder_output

        # 2. Prepare encoder args and encoder kwargs from model kwargs.
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        # encoder_signature = set(inspect.signature(encoder.forward).parameters)
        encoder_signature = set(inspect.signature(encoder).parameters)

        encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        if not encoder_accepts_wildcard:
            encoder_kwargs = {
                argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature
            }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        model_kwargs["encoder_outputs"] = encoder(**encoder_kwargs)
        # model_kwargs["encoder_outputs"] = self.get_encoder_output(**encoder_kwargs)

        # dict_keys(['attention_mask', 'ent_indices', 'ent_mask', 'output_attentions', 'output_hidden_states', 'use_cache', 'encoder_outputs'])
        return model_kwargs


class FiD(pt.Transformer):
    def __init__(
        self,
        model: Union[T5FiDReader, BARTFiDReader],
        tokenizer: AutoTokenizer,
        batch_size: int = 4,
        text_field: str = 'text',
        text_max_length: int = 256,
        num_context: Union[int, str] = "auto",
        max_new_tokens: int = 32,
        generation_config: GenerationConfig = None,
        verbose: bool = False,
        device: Union[str, torch.device] = None,
        **kwargs
    ):
        self.model = model.to(device)
        self.model.eval()
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.text_field = text_field
        self.text_max_length = text_max_length
        self.num_context = num_context
        self.max_new_tokens = max_new_tokens
        self.generation_config = generation_config
        self.device = device
        self.query_prefix = "question:"
        self.title_prefix = "title:"
        self.context_prefix = "context:"
        self.verbose = verbose
    
    def get_context_by_query(self, inp: Iterable[dict]) -> Iterable[Union[str, Tuple[str]]]:
        """Return at most self.num_context retrieved context.
        """
        if self.num_context and inp:
            num = len(inp) if self.num_context == "auto" else self.num_context
            if "score" in inp[0]:
                inp = sorted(inp, key=lambda x: x["score"], reverse=True)
            if "title" in inp[0]:
                context = [(item["title"], item[self.text_field]) for item in inp]
            else:
                context = [item[self.text_field] for item in inp]
            context = context[:num]
        else:
            context = None
        return context

    def format_input_texts(self, question: str, context: Iterable[Union[str, Tuple[str]]]) -> List[str]:

        if not context:
            return [question]

        input_texts = []
        for item in context:
            # append title and context prefix
            if isinstance(item, tuple):
                title, text = item
                doc_text = self.title_prefix + " " + title + " " + self.context_prefix + " " + text
            else:
                text = item
                doc_text = self.context_prefix + " " + text
            # prepend question
            input_text = self.query_prefix + " " + question + " " + doc_text.strip()
            input_texts.append(input_text.strip())

        return input_texts

    def tokenizer_encode(self, texts: List[str]) -> Dict[str, torch.Tensor]:

        tokenizer_outputs = self.tokenizer.batch_encode_plus(
            texts,
            max_length = self.text_max_length,
            padding = "max_length",
            truncation = True,
            return_tensors = 'pt'
        )
        input_ids = tokenizer_outputs["input_ids"][None, :, :] # for only one query
        attention_mask = tokenizer_outputs["attention_mask"][None, :, :]

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    @pta.transform.by_query(add_ranks=False)
    def transform_iter(self, inp: Iterable[dict]) -> Iterable[dict]:
        return self.transform_by_query(inp)

    def transform_by_query(self, inp: Iterable[dict]) -> Iterable[dict]:
        inp = list(inp)
        qid = inp[0]["qid"]
        query = inp[0]["query"]
        for row in inp:
            assert row["query"] == query, "All rows must have the same query for `transform_by_query`"

        context = self.get_context_by_query(inp)
        input_texts = self.format_input_texts(query, context)
        inputs = self.tokenizer_encode(input_texts)
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        generated_token_ids = self.model.generate(**inputs, generation_config=self.generation_config)
        qanswer = self.tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)[0]

        return [ {"qid": qid, "query": query, "qanswer": qanswer} ]


class T5FiD(FiD):

    def __init__(self, model_name_or_path: str, tokenizer_name_or_path: str = None, batch_size: int = 4, text_field: str = 'text', text_max_length: int = 256, num_context: Union[int, str] = "auto", max_new_tokens: int = 32, generation_config: GenerationConfig = None, verbose: bool = False, device: Union[str, torch.device] = None, **kwargs):
        model = T5FiDReader.from_pretrained(model_name_or_path)
        tokenizer_name_or_path = tokenizer_name_or_path or model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        super().__init__(model, tokenizer, batch_size, text_field, text_max_length, num_context, max_new_tokens, generation_config, verbose, model.device, **kwargs)


class BARTFiD(FiD):

    def __init__(self, model_name_or_path: str, tokenizer_name_or_path: str = None, batch_size: int = 4, text_field: str = 'text', text_max_length: int = 256, num_context: Union[int, str] = "auto", max_new_tokens: int = 32, generation_config: GenerationConfig = None, verbose: bool = False, device: Union[str, torch.device] = None, **kwargs):
        model = BARTFiDReader.from_pretrained(model_name_or_path)
        tokenizer_name_or_path = tokenizer_name_or_path or model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        super().__init__(model, tokenizer, batch_size, text_field, text_max_length, num_context, max_new_tokens, generation_config, verbose, model.device, **kwargs)
