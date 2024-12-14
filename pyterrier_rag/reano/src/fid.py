
import inspect 
import logging 

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import AutoTokenizer
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput
# from transformers.models.t5.configuration_t5 import T5Config

logger = logging.getLogger(__name__)


class TripleRelationExtraction(nn.Module):

    def __init__(self, input_dim, ent_dim, rel_dim, dropout=0.25, k=5):

        super().__init__()

        if input_dim != ent_dim:
            self.ent_project = nn.Sequential(
                nn.Linear(input_dim, ent_dim * 2),
                nn.GELU(),
                nn.Linear(ent_dim * 2, ent_dim), 
                nn.GELU(),
                nn.Dropout(0.1),
            )

        self.rel_project = nn.Sequential(
            nn.Linear(ent_dim * 2, ent_dim * 2), 
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(ent_dim * 2, rel_dim),
            nn.GELU()
        )

        self.input_dim = input_dim
        self.ent_dim = ent_dim
        self.rel_dim = rel_dim
        self.k = k 
    
    def get_pooled_question_embedding(self, question_embedding, question_mask):

        expand_question_mask = question_mask.unsqueeze(-1)
        q = torch.sum(question_embedding * expand_question_mask, dim=1) / torch.sum(expand_question_mask, dim=1)
        return q 
    
    def forward(self, hidden_states, ent_indices, ent_mask, adj=None, adj_mask=None, adj_relation_embedding=None):

        batch_size, num_entity_tokens, hidden_size = hidden_states.shape 
        batch_size, max_num_entity, max_num_mention = ent_indices.shape 
        flatten_ent_indices = ent_indices.reshape(batch_size, -1)
        entity_span_embedding = hidden_states.gather(1, flatten_ent_indices.unsqueeze(-1).expand(-1, -1, hidden_size))
        entity_span_embedding = entity_span_embedding.reshape(batch_size, max_num_entity, max_num_mention, hidden_size)

        if hasattr(self, "ent_project"):
            entity_span_embedding = self.ent_project(entity_span_embedding)

        expand_ent_mask = ent_mask.unsqueeze(-1) 
        ent_num_mention = torch.sum(expand_ent_mask, dim=-2)
        ent_num_mention[ent_num_mention == 0] = 1 
        entity_embedding = torch.sum(entity_span_embedding * expand_ent_mask, dim=-2) / ent_num_mention

        node_mask = torch.sum(ent_mask, dim=-1) > 0 

        source_node = torch.arange(max_num_entity, device=adj.device)[None, :, None].expand(batch_size, -1, adj.shape[-1]).reshape(batch_size, -1)
        target_node = adj.reshape(batch_size, -1)
        source_node_embedding = entity_embedding.gather(1, source_node.unsqueeze(-1).expand(-1, -1, self.ent_dim))
        target_node_embedding = entity_embedding.gather(1, target_node.unsqueeze(-1).expand(-1, -1, self.ent_dim))

        rel_embedding = self.rel_project(torch.cat((source_node_embedding, target_node_embedding), dim=-1))
        rel_embedding_size = rel_embedding.shape[-1]
        if adj_relation_embedding is not None:
            rel_embedding += adj_relation_embedding.reshape(batch_size, -1, rel_embedding_size)
        
        # rel_embedding = rel_embedding.reshape(batch_size, max_num_entity, self.k, rel_embedding_size)
        rel_embedding = rel_embedding.reshape(batch_size, max_num_entity, -1, rel_embedding_size)

        return entity_embedding, rel_embedding, node_mask


class TripleGraphNeuralNet(nn.Module):

    def __init__(self, ent_dim, rel_dim, hidden_dim, hop=2, dropout=0.1) -> None:

        super().__init__()

        self.ent_dim = ent_dim 
        self.rel_dim = rel_dim 
        self.hidden_dim = hidden_dim 
        self.hop = hop 

        self.projection = nn.Linear(ent_dim, hidden_dim)
        self.relation_projection = nn.Linear(rel_dim, hidden_dim)

        self.message_projection = nn.ModuleList([nn.Linear(self.hidden_dim*2, self.hidden_dim) for _ in range(self.hop)])
        self.entity_update = nn.ModuleList([nn.Linear(self.hidden_dim*2, self.hidden_dim) for _ in range(self.hop)])
        self.rel_score = nn.Linear(self.hidden_dim*2, 1)
        self.score = nn.Linear(self.hidden_dim, 1) 
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    
    def get_pooled_question_embedding(self, question_embedding, question_mask):

        expand_question_mask = question_mask.unsqueeze(-1)
        q = torch.sum(question_embedding * expand_question_mask, dim=1) / torch.sum(expand_question_mask, dim=1)
        return q 
    
    def forward(self, question_embedding, question_mask, entity_embedding, rel_embedding, adj, node_mask, adj_mask):

        question_embedding = self.projection(question_embedding)
        entity_embedding = self.projection(entity_embedding)
        rel_embedding = self.relation_projection(rel_embedding)
        q = self.get_pooled_question_embedding(question_embedding, question_mask) 
        batch_size, max_num_entity, num_rel_per_entity, hidden_dim = rel_embedding.shape 

        dtype = rel_embedding.dtype
        weight_mask = (1.0 - adj_mask.type(dtype)) * torch.finfo(dtype).min 
        expand_q = q[:, None, None, :].expand(-1, max_num_entity, num_rel_per_entity, -1)
        weight_logit = self.rel_score(torch.cat([rel_embedding, expand_q], dim=-1)).squeeze(-1) + weight_mask
        weight = adj_mask * F.softmax(weight_logit, dim=-1) 
        
        for k in range(self.hop):
            tail_entity_embedding = entity_embedding.gather(1, adj.reshape(batch_size, -1)[:, :, None].expand(-1, -1, hidden_dim))
            tail_entity_embedding = tail_entity_embedding.reshape(batch_size, max_num_entity, num_rel_per_entity, hidden_dim)
            messages = self.message_projection[k](torch.cat([rel_embedding, tail_entity_embedding], dim=-1)) 
            messages = self.dropout(self.act(messages))
            aggregation = torch.sum(messages * weight[:, :, :, None], dim=-2) 
            entity_embedding = self.entity_update[k](torch.cat([entity_embedding, aggregation], dim=-1))
            entity_embedding = self.dropout(self.act(entity_embedding))

        extended_node_mask = (1.0 - node_mask.unsqueeze(-1).type(dtype)) * torch.finfo(dtype).min
        ent_score = (self.score(entity_embedding) + extended_node_mask).squeeze(-1)

        with torch.no_grad():
            source_node = torch.arange(max_num_entity, device=adj.device)[None, :, None].expand(batch_size, -1, adj.shape[-1]).reshape(batch_size, -1)
            target_node = adj.reshape(batch_size, -1)
            source_node_embedding = entity_embedding.gather(1, source_node.unsqueeze(-1).expand(-1, -1, self.hidden_dim))
            target_node_embedding = entity_embedding.gather(1, target_node.unsqueeze(-1).expand(-1, -1, self.hidden_dim))
            q_source_sim = torch.matmul(q.unsqueeze(1), torch.transpose(source_node_embedding, 1, 2)).squeeze(1)
            q_target_sim = torch.matmul(q.unsqueeze(1), torch.transpose(target_node_embedding, 1, 2)).squeeze(1)
            triple_sim = q_source_sim + q_target_sim + weight_logit.reshape(batch_size, -1)
        
        return entity_embedding, ent_score, weight_logit, triple_sim


class TripleKGFiDT5(T5ForConditionalGeneration):

    def __init__(self, config, tokenizer: AutoTokenizer=None, ent_dim: int=128, k: int = 5, hop: int = 2, alpha: int = 1.0, num_triples: int=20):

        super().__init__(config)
        
        self.tokenizer = tokenizer
        self.ent_dim = self.model_dim
        self.k = k
        self.hop = hop 
        self.alpha = alpha 
        self.num_triples = num_triples
        self.word_embeddings = self.shared

        self.relation_extraction = TripleRelationExtraction(self.model_dim, self.ent_dim, self.ent_dim, k=self.k)
        self.gnn = TripleGraphNeuralNet(ent_dim=self.ent_dim, rel_dim=self.ent_dim, hidden_dim=256, hop=self.hop)

        self.init_weights()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            if module.elementwise_affine == True:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def relation_extraction_setup(self, relationid2name, relation_embedding):

        self.relationid2name = relationid2name 
        num_relation, embedding_size = relation_embedding.size()
        self.relation_pooled_token_embedding = nn.Embedding(num_relation, embedding_size)
        self.relation_pooled_token_embedding.weight = nn.Parameter(relation_embedding, requires_grad=False)

    def get_question_embedding(self, hidden_states, question_indices, question_mask, batch_size):

        B_hidden_states, num_entity_tokens, hidden_size = hidden_states.shape 
        B, max_question_len = question_indices.shape 
        assert B_hidden_states == B 
        question_embedding = hidden_states.gather(1, question_indices.unsqueeze(-1).expand(-1, -1, hidden_size)) 

        question_embedding = question_embedding.reshape(batch_size, B // batch_size, max_question_len, -1)
        question_mask = question_mask.reshape(batch_size, B // batch_size, max_question_len)
        question_embedding = torch.mean(question_embedding, dim=1) 
        question_mask = torch.sum(question_mask, dim=1) > 0

        return question_embedding, question_mask

    def scatter_entity_embedding(self, hidden_states, entity_embedding, ent_indices, ent_mask):

        batch_size, num_tokens, hidden_size = hidden_states.shape 
        batch_size, max_num_entity, max_num_mention = ent_indices.shape 
        entity_embedding = self.entity_project(entity_embedding) 
        expand_entity_embedding = entity_embedding.unsqueeze(-2).expand(-1, -1, max_num_mention, -1) 
        expand_entity_embedding = expand_entity_embedding * ent_mask.unsqueeze(-1)

        expand_entity_embedding = expand_entity_embedding.reshape(batch_size, -1, hidden_size)
        flatten_ent_indices = ent_indices.reshape(batch_size, -1)
        update_hidden_states = torch.zeros_like(hidden_states)
        update_hidden_states.scatter_(1, flatten_ent_indices.unsqueeze(-1).expand(-1, -1, hidden_size), expand_entity_embedding)
        hidden_states = hidden_states + update_hidden_states

        return hidden_states

    def get_pooled_embeddings(self, embeddings, attention_mask):

        expanded_attention_mask = attention_mask[:, :, None]
        embeddings = torch.sum(embeddings * expanded_attention_mask, dim=1)
        return embeddings / torch.sum(expanded_attention_mask, dim=1)
    
    def get_entity_adj_relation_embedding(self, entity_adj_relation):

        batch_size, num_entity, num_neighbor = entity_adj_relation.shape 
        entity_adj_relation_embedding = self.relation_pooled_token_embedding(entity_adj_relation.reshape(-1))
        entity_adj_relation_embedding = entity_adj_relation_embedding.reshape(batch_size, num_entity, num_neighbor, -1)

        return entity_adj_relation_embedding

    def get_encoder_output(
        self, 
        input_ids, 
        attention_mask,
        question_indices=None,
        question_mask=None,
        ent_indices=None,
        ent_mask=None,
        entity_text=None, 
        entity_adj=None, 
        entity_adj_mask=None, 
        entity_adj_relation=None, 
        entity_adj_relevant_relation_label=None, 
        mask_passages=False, 
        num_passages_after_mask=None, 
        passage_entity_ids=None, 
        passage_entity_mask=None,
        output_attentions=False, 
        output_hidden_states=False, 
        **kwargs
    ):

        batch_size, num_passages, seq_len = input_ids.shape 
        input_ids = input_ids.reshape(-1, seq_len) 
        attention_mask = attention_mask.reshape(-1, seq_len)

        encoder_outputs = self.encoder(
            input_ids = input_ids, 
            attention_mask = attention_mask, 
            output_attentions = output_attentions, 
            output_hidden_states = output_hidden_states, 
            return_dict = False, 
        )

        hidden_states = encoder_outputs[0] 
        hidden_size = hidden_states.shape[-1]
        device = hidden_states.device

        question_embedding, question_mask = self.get_question_embedding(hidden_states, question_indices, question_mask, batch_size=batch_size)

        hidden_states = hidden_states.reshape(batch_size, -1, hidden_size)
        
        entity_adj_relation_embedding = self.get_entity_adj_relation_embedding(entity_adj_relation)
        entity_embedding, rel_embedding, node_mask = self.relation_extraction(
            hidden_states, ent_indices, ent_mask, entity_adj, entity_adj_mask, entity_adj_relation_embedding
        )

        updated_entity_embedding, entity_score, adj_relation_score, triple_sim = self.gnn(question_embedding, question_mask, entity_embedding, rel_embedding, entity_adj, node_mask, entity_adj_mask)

        with torch.no_grad():
            max_num_entity = entity_adj.shape[1]
            num_topk_rel = self.num_triples 
            max_triple_seq_len = 512 
            source_node = torch.arange(max_num_entity, device=device)[None, :, None].expand(batch_size, -1, entity_adj.shape[-1]).reshape(batch_size, -1)
            target_node = entity_adj.reshape(batch_size, -1)
            topk_triple_scores, topk_triple_indices = torch.topk(triple_sim, k = num_topk_rel, dim=-1)
            topk_source_node = source_node.gather(1, topk_triple_indices).tolist()
            topk_target_node = target_node.gather(1, topk_triple_indices).tolist()
            topk_relation = entity_adj_relation.reshape(batch_size, -1).gather(1, topk_triple_indices).tolist()

            triple_text_list = []
            min_value = torch.finfo(rel_embedding.dtype).min 
            batch_question_text = kwargs.pop("question_text", None)
            for i in range(batch_size):
                text = "Some relevant knowledge triples: " if batch_question_text is None else "{} context:".format(batch_question_text[i])
                for j, (s, t, r) in enumerate(zip(topk_source_node[i], topk_target_node[i], topk_relation[i])):
                    if topk_triple_scores[i, j] == min_value:
                        break
                    text = text + "<e> {} </e> {} <e> {} </e>; ".format(entity_text[i][s], self.relationid2name[r], entity_text[i][t])
                triple_text_list.append(text)

            triple_inputs = self.tokenizer(triple_text_list, padding=True, truncation=True, max_length=max_triple_seq_len, return_tensors='pt')
            triple_input_ids = triple_inputs["input_ids"].to(device)
            triple_attention_mask = triple_inputs["attention_mask"].to(device)
        
        triple_hidden_states = self.encoder(
            input_ids = triple_input_ids, 
            attention_mask = triple_attention_mask,
            output_attentions = output_attentions, 
            output_hidden_states = output_hidden_states, 
            return_dict = False, 
        )[0]

        if not mask_passages:
            passage_hidden_states = hidden_states
            passage_attention_mask = attention_mask.reshape(batch_size, -1)
        else:
            assert num_passages_after_mask is not None 
            assert passage_entity_ids is not None 
            assert passage_entity_mask is not None 

            with torch.no_grad():
                entity_prob = F.softmax(entity_score, dim=1)
                passage_entity_prob = entity_prob.gather(1, passage_entity_ids.reshape(batch_size, -1)).reshape(passage_entity_ids.size())
                dtype = passage_entity_prob.dtype 
                passage_entity_prob = passage_entity_prob + (1.0 - passage_entity_mask.to(dtype)) * torch.finfo(dtype).min
                passage_max_entity_prob = torch.max(passage_entity_prob, dim=-1)[0]
                topk_passage_indices = torch.topk(passage_max_entity_prob, k=num_passages_after_mask)[1] 
                topk_passage_hidden_state_indices = torch.zeros((batch_size, num_passages_after_mask*seq_len), dtype=torch.long)
                for i in range(batch_size):
                    for j, passage_idx in enumerate(topk_passage_indices[i]):
                        topk_passage_hidden_state_indices[i, j*seq_len: (j+1)*seq_len] = torch.arange(passage_idx*seq_len, (passage_idx+1)*seq_len)

            orig_passage_hidden_states = hidden_states
            orig_passage_attention_mask = attention_mask.reshape(batch_size, -1)
            topk_passage_hidden_state_indices = topk_passage_hidden_state_indices.to(device)
            passage_hidden_states = orig_passage_hidden_states.gather(1, topk_passage_hidden_state_indices[:, :, None].expand(-1, -1, hidden_size))
            passage_attention_mask = orig_passage_attention_mask.gather(1, topk_passage_hidden_state_indices)

        updated_question_ent_hidden_states = torch.cat([passage_hidden_states, triple_hidden_states], dim=1)
        question_ent_mask = torch.cat([passage_attention_mask, triple_attention_mask], dim=1)
        
        return {
            "question_ent_hidden_states": updated_question_ent_hidden_states, 
            "question_ent_mask": question_ent_mask, 
            "node_mask": node_mask, 
            "entity_score": entity_score, 
            "adj_relation_score": adj_relation_score,
        }

    def forward(
        self, 
        input_ids=None, 
        attention_mask=None, 
        decoder_input_ids=None, 
        decoder_attention_mask=None, 
        encoder_outputs=None, 
        question_indices=None,
        question_mask=None,
        ent_indices=None,
        ent_mask=None, 
        entity_text=None, 
        entity_adj=None, 
        entity_adj_mask=None,
        entity_adj_relation=None, 
        labels=None, 
        entity_adj_relevant_relation_label=None, 
        ent_is_ans_label=None,
        mask_passages=False, 
        num_passages_after_mask=None, 
        passage_entity_ids=None, 
        passage_entity_mask=None,
        output_attentions=False, 
        output_hidden_states=False,
        return_dict=False,
        **kwargs
    ):
        
        calculate_kg_loss = kwargs.get("calculate_kg_loss", False)
        calculate_ans_loss = kwargs.get("calculate_ans_loss", labels is not None)

        if encoder_outputs is None:
            encoder_outputs = self.get_encoder_output(
                input_ids = input_ids, 
                attention_mask = attention_mask, 
                question_indices=question_indices, 
                question_mask=question_mask,
                ent_indices=ent_indices, 
                ent_mask=ent_mask, 
                entity_text = entity_text, 
                entity_adj=entity_adj, 
                entity_adj_mask=entity_adj_mask, 
                entity_adj_relation=entity_adj_relation, 
                entity_adj_relevant_relation_label=entity_adj_relevant_relation_label,
                mask_passages=mask_passages, 
                num_passages_after_mask=num_passages_after_mask, 
                passage_entity_ids=passage_entity_ids, 
                passage_entity_mask=passage_entity_mask,
                output_attentions = output_attentions, 
                output_hidden_states = output_hidden_states,
                **kwargs
            )

        question_ent_hidden_states = encoder_outputs["question_ent_hidden_states"]
        question_ent_mask = encoder_outputs["question_ent_mask"]
        node_mask = encoder_outputs["node_mask"]
        entity_score = encoder_outputs["entity_score"]
        adj_relation_score = encoder_outputs["adj_relation_score"]

        if calculate_kg_loss:
            
            ent_is_ans_label = ent_is_ans_label.to(entity_score.dtype).to(entity_score.device)
            ent_is_ans_label[(ent_is_ans_label > 0.0) & (~node_mask)] = 0.0 
            ent_loss_fct = nn.KLDivLoss(reduction='none')
            answer_len = torch.sum(ent_is_ans_label, dim=1, keepdim=True)
            answer_len[answer_len == 0] = 1.0 
            answer_prob = ent_is_ans_label.div(answer_len)
            log_ent_score = torch.log(F.softmax(entity_score, dim=1) + 1e-9).squeeze(-1)
            ent_loss = ent_loss_fct(log_ent_score, answer_prob).sum(dim=1) 
            has_ans_mask = torch.sum(ent_is_ans_label, dim=1) > 0 
            has_ans_num = torch.sum(has_ans_mask)
            if has_ans_num > 0:
                ent_loss = torch.sum(ent_loss * has_ans_mask) / has_ans_num
            else:
                ent_loss = torch.sum(ent_loss * has_ans_mask)
            
            kg_loss = self.alpha * ent_loss
            
            mask = entity_adj_mask & (torch.sum(entity_adj_relevant_relation_label, dim=(1, 2)) > 0)[:, None, None]
            filter_relation_score = adj_relation_score[mask]
            filter_relation_label = entity_adj_relevant_relation_label[mask]
            if len(filter_relation_score) > 0:
                pos_weight = torch.sum(filter_relation_label==0) / torch.sum(filter_relation_label==1)
                relation_loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                relation_loss = relation_loss_fct(filter_relation_score, filter_relation_label.to(filter_relation_score))
                kg_loss += self.alpha * relation_loss

        if calculate_kg_loss and not calculate_ans_loss:
            return (kg_loss, )

        if labels is not None:
            decoder_input_ids = self._shift_right(labels)

        decoder_output = self.decoder(
            input_ids = decoder_input_ids, 
            attention_mask = decoder_attention_mask, 
            encoder_hidden_states = question_ent_hidden_states,
            encoder_attention_mask = question_ent_mask, 
            output_attentions = output_attentions, 
            output_hidden_states = output_hidden_states, 
            return_dict = False,
        )

        sequence_output = decoder_output[0]
        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim**-0.5)
        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            labels = labels.to(lm_logits.device)
            ans_loss = loss_fct(lm_logits.reshape(-1, lm_logits.shape[-1]), labels.reshape(-1))
        else:
            ans_loss = None
        
        if not calculate_kg_loss and calculate_ans_loss:
            loss = ans_loss

        if calculate_kg_loss and calculate_ans_loss:
            loss = kg_loss + ans_loss
        
        if not return_dict:
            output = (lm_logits, question_ent_hidden_states)
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
        input_ids, 
        attention_mask = None, 
        question_indices = None,
        question_mask = None, 
        ent_indices = None, 
        ent_mask = None,
        entity_text=None, 
        entity_adj=None, 
        entity_adj_mask=None, 
        entity_adj_relation=None, 
        question_text=None, 
        mask_passages=False, 
        num_passages_after_mask=None, 
        passage_entity_ids=None, 
        passage_entity_mask=None, 
        decoder_attention_mask = None, 
        encoder_outputs = None,
        **kwargs, 
    ):        
        return {
            "encoder_outputs": encoder_outputs,
            "decoder_input_ids": input_ids, 
            "decoder_attention_mask": decoder_attention_mask,
        }
