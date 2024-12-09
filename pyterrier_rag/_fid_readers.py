import inspect
from typing import Optional, Dict, List

import torch
import torch.nn as nn 
from torch.nn import CrossEntropyLoss
from transformers import T5ForConditionalGeneration, BartModel, BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput, Seq2SeqModelOutput
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.bart.configuration_bart import BartConfig


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
            return_dict = False, 
        )

        hidden_states = encoder_outputs[0] # batch_size x num_passages, seq_length, hidden_size 
        hidden_size = hidden_states.shape[-1]
        if need_flatten:
            hidden_states = hidden_states.reshape(batch_size, num_passages*seq_length, hidden_size)
            attention_mask = attention_mask.reshape(batch_size, num_passages*seq_length)

        return {
            "hidden_states": hidden_states, 
            "attention_mask": attention_mask
        }

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
        
        encoder_hidden_states = encoder_outputs["hidden_states"]
        encoder_attention_mask = encoder_outputs["attention_mask"]

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
    """
    Shift input ids one token to the right.
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
        
        return BaseModelOutput(
            last_hidden_state=hidden_states, 
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
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask.reshape(attention_mask.shape[0], -1),
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
    