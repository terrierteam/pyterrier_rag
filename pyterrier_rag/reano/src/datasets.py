
import pickle
import random
import logging

import torch
from torch.utils.data import Dataset

from src.evaluation import ems 
from src.util import remove_bracket

logger = logging.getLogger(__name__)


class TripleFiDDataset(Dataset):

    def __init__(self, data_path, n_context=None, question_prefix='question:', title_prefix='title:', passage_prefix='context:', max_num_entities=2000):

        self.data = self.load_data(data_path)
        self.n_context = n_context
        self.question_prefix = question_prefix
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix
        self.max_num_entities = max_num_entities
        self.sort_data()
        self.use_entityname = "entityquestion" in data_path

    def load_data(self, data_path):
        
        print("Loading data from {} ... ".format(data_path))
        data = pickle.load(open(data_path, "rb"))
        examples = [] 
        for k, example in enumerate(data):
            if not 'id' in example:
                example['id'] = k 
            for c in example['ctxs']:
                if not 'score' in c:
                    c['score'] = 1.0 
            examples.append(example)

        return examples 

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        if 'target' in example:
            target = example['target']
            return target + ' </s>'
        elif 'answers' in example:
            return random.choice(example['answers']) + ' </s>'
        else:
            return None

    def get_context_text(self, text_with_entity, entity_list, batch_ent_map, batch_entity_name, batch_ent_is_answer, answers, entityid2name):

        entity_type = []
        sorted_entity_list = sorted(entity_list, key=lambda x: x[0])
        for start_idx, end_idx, mention, entity_id in sorted_entity_list:

            if entity_id not in batch_ent_map:
                batch_ent_map[entity_id] = len(batch_ent_map)
                if not self.use_entityname:
                    entity_name = remove_bracket(mention)
                else:
                    entity_name = remove_bracket(entityid2name[entity_id][0])
                batch_entity_name.append(entity_name)
                batch_ent_is_answer[batch_ent_map[entity_id]] = self.is_answer(entity_name, answers)

            entity_type.append(batch_ent_map[entity_id])

        return text_with_entity, entity_type
    
    def is_answer(self, text, answers):
        return ems(text, answers)

    def __getitem__(self, index):

        example = self.data[index]
        question = self.question_prefix + " " + example['question']
        gold_answers = example["answers"] 
        target = self.get_target(example)
        entityid2name = example["entityid2name"]

        batch_ent_map = {}
        batch_entity_name = []
        batch_ent_is_answer = {}

        question_entity_list = [] 
        for i, question_entity in enumerate(example["question_entity"]):
            batch_ent_map[i] = len(batch_ent_map)
            entity_name = remove_bracket(question_entity)
            batch_entity_name.append(entity_name)
            batch_ent_is_answer[batch_ent_map[i]] = self.is_answer(entity_name, answers=gold_answers)
            question_entity_list.append(batch_ent_map[i])
        
        if 'ctxs' in example and self.n_context is not None:

            f = self.title_prefix + " {} " + self.passage_prefix + " {}"
            contexts = example['ctxs'][:self.n_context] 

            passages, entity_type_list = [], []
            for c in contexts:
                title, title_entity = self.get_context_text(c["title"], c["title_entity"], batch_ent_map, batch_entity_name, batch_ent_is_answer, gold_answers, entityid2name)
                text, text_entity = self.get_context_text(c["text"], c["text_entity"], batch_ent_map, batch_entity_name, batch_ent_is_answer, gold_answers, entityid2name)
                passages.append(f.format(title, text))
                entity_type_list.append(title_entity + text_entity)

            scores = [float(c['score']) for c in contexts]
            scores = torch.tensor(scores)

            if len(passages) < self.n_context:
                while len(passages) < self.n_context:
                    passages = passages + passages[:(self.n_context-len(passages))]
                    entity_type_list = entity_type_list + entity_type_list[:(self.n_context-len(entity_type_list))]

        else:
            passages, scores, entity_type_list = None, None, None

        num_entity = len(batch_ent_map)
        
        triples = []
        for heid, rel, teid in example.get("triples", []): 
            if heid in batch_ent_map and teid in batch_ent_map:
                triple = (batch_ent_map[heid], rel, batch_ent_map[teid])
                if triple not in triples:
                    triples.append(triple)

        entity_is_answer_list = [batch_ent_is_answer[i] for i in range(len(batch_entity_name))] 
        relevant_triples = []
        for heid, rel, teid in example.get("relevant_triples", []):
            if heid in batch_ent_map and teid in batch_ent_map:
                triple = (batch_ent_map[heid], rel, batch_ent_map[teid])
                entity_is_answer_list[batch_ent_map[heid]] = True
                entity_is_answer_list[batch_ent_map[teid]] = True
                if triple not in relevant_triples:
                    relevant_triples.append(triple)

        return {
            'index' : index,
            'question' : question,
            'target' : target,
            'passages' : passages,
            'scores' : scores,
            "num_entity": num_entity, 
            'entity': batch_entity_name,
            'entity_type_list': entity_type_list,
            "entity_is_answer_list": entity_is_answer_list,
            "triples": triples, 
            "relevant_triples": relevant_triples,
            "evidences": example.get("evidences", None), 
            "question_entity": question_entity_list, 
        }

    def sort_data(self):
        if self.n_context is None or not 'score' in self.data[0]['ctxs'][0]:
            return
        for ex in self.data:
            ex['ctxs'].sort(key=lambda x: float(x['score']), reverse=True)

    def get_example(self, index):
        return self.data[index]


def encode_passages(batch_text_passages, tokenizer, max_length):

    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=max_length,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True
        )
        passage_ids.append(p['input_ids'][None])
        passage_masks.append(p['attention_mask'][None])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)

    return passage_ids, passage_masks.bool()


def encode_entities(batch_entities, tokenizer, batch_max_num_entities):

    all_entities = [] 

    for entities in batch_entities:
        # padding 
        pad_entities = entities + [""] * (batch_max_num_entities - len(entities))
        all_entities.extend(pad_entities)

    outputs = tokenizer.batch_encode_plus(
        all_entities, 
        padding=True, 
        truncation=True,
        max_length=16,
        return_tensors='pt',
        add_special_tokens=False, # NOTE: entity的id不添加eos_token 
    )

    batch_size = len(batch_entities)
    entity_input_ids = outputs["input_ids"].reshape(batch_size, batch_max_num_entities, -1)
    entity_attention_mask = outputs["attention_mask"].reshape(batch_size, batch_max_num_entities, -1)

    return entity_input_ids, entity_attention_mask.bool()


class FiDCollator(object):

    def __init__(self, text_maxlength, tokenizer, relation2id, answer_maxlength=20, max_num_entities=2000, max_num_mention_per_entity=50, max_num_edge=25):

        self.tokenizer = tokenizer
        self.relation2id = relation2id
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength
        self.max_num_entities = max_num_entities
        self.max_num_mention_per_entity = max_num_mention_per_entity
        self.max_num_edge = max_num_edge

        self.sep_token_id = self.tokenizer.encode("[SEP]")[0]
        self.cls_token_id = self.tokenizer.encode("[CLS]")[0]
        self.ent_start_id = self.tokenizer.encode("<e>")[0]
        self.ent_end_id = self.tokenizer.encode("</e>")[0]

    def __call__(self, batch):

        assert(batch[0]['target'] != None)
        index = torch.tensor([ex['index'] for ex in batch])
        target = [ex['target'] for ex in batch]
        target = self.tokenizer.batch_encode_plus(
            target,
            max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True if self.answer_maxlength > 0 else False,
        )
        target_ids = target["input_ids"]
        target_mask = target["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)
        
        def append_question(example):
            if example['passages'] is None:
                return [self.maybe_truncate_question(example['question'])]
            return [self.maybe_truncate_question(example['question']) + " [SEP] " + t for t in example['passages']]
        
        text_passages = [append_question(example) for example in batch]

        passage_ids, passage_masks = encode_passages(text_passages, self.tokenizer, self.text_maxlength) 
        
        batch_size, num_passages = passage_ids.shape[0], passage_ids.shape[1]
        flatten_passage_ids = passage_ids.reshape(-1, self.text_maxlength)
        B, maxlen = flatten_passage_ids.shape

        batch_question_text = [self.maybe_truncate_question(example["question"]) for example in batch]
        indices = torch.arange(maxlen)[None, :].expand(B, -1)
        question_length = (flatten_passage_ids == self.sep_token_id).nonzero()[:, -1:]
        question_mask = indices < question_length # B x maxlen 

        ent_start_row_indices, ent_start_col_indices = (flatten_passage_ids == self.ent_start_id).nonzero(as_tuple=True)
        ent_end_row_indices, ent_end_col_indices = (flatten_passage_ids == self.ent_end_id).nonzero(as_tuple=True)

        batch_max_num_entities = min(max([example["num_entity"] for example in batch]), self.max_num_entities)
        batch_entity_type_list = [example["entity_type_list"] for example in batch]
        batch_entity_num_mention = torch.zeros((batch_size, batch_max_num_entities), dtype=torch.long)
        batch_entity_mention_indices = torch.zeros((batch_size, batch_max_num_entities, self.max_num_mention_per_entity), dtype=torch.long)
        batch_entity_mention_passage_indices = torch.zeros((batch_size, batch_max_num_entities, self.max_num_mention_per_entity), dtype=torch.long)
        batch_entity_mention_mask = torch.zeros((batch_size, batch_max_num_entities, self.max_num_mention_per_entity), dtype=torch.bool)

        max_num_entity_per_passage = 25 
        batch_passage_entity_length = torch.zeros((batch_size, num_passages), dtype=torch.long)
        batch_passage_entity_ids = torch.zeros((batch_size, num_passages, max_num_entity_per_passage), dtype=torch.long)
        batch_passage_entity_mask = torch.zeros((batch_size, num_passages, max_num_entity_per_passage), dtype=torch.bool)

        max_num_edge_per_entity = self.max_num_edge
        batch_entity_adj = torch.zeros((batch_size, batch_max_num_entities, max_num_edge_per_entity), dtype=torch.long)
        batch_entity_num_edge = torch.zeros((batch_size, batch_max_num_entities), dtype=torch.long)
        batch_entity_adj_mask = torch.zeros((batch_size, batch_max_num_entities, max_num_edge_per_entity), dtype=torch.bool)
        batch_entity_adj_relation = torch.zeros((batch_size, batch_max_num_entities, max_num_edge_per_entity), dtype=torch.long) 
        batch_entity_adj_relevant_relation_label = torch.zeros((batch_size, batch_max_num_entities, max_num_edge_per_entity), dtype=torch.long) 
        batch_triples = [example["triples"] for example in batch]
        batch_relevant_triples = [example["relevant_triples"] for example in batch] 

        batch_has_mention_entities = [set() for i in range(batch_size)] 

        for i in range(B):

            batch_idx, passage_idx = i // num_passages, i % num_passages

            for ent_start_idx, ent_end_idx, ent_type in zip(
                ent_start_col_indices[ent_start_row_indices==i],
                ent_end_col_indices[ent_end_row_indices==i], 
                batch_entity_type_list[batch_idx][passage_idx]
            ):
                
                if ent_type >= batch_max_num_entities:
                    continue
                num_existing_mention = batch_entity_num_mention[batch_idx, ent_type]
                if num_existing_mention >= self.max_num_mention_per_entity:
                    continue

                batch_entity_mention_indices[batch_idx, ent_type, num_existing_mention] = ent_start_idx
                batch_entity_mention_passage_indices[batch_idx, ent_type, num_existing_mention] = passage_idx 
                batch_entity_mention_mask[batch_idx, ent_type, num_existing_mention] = True 
                batch_entity_num_mention[batch_idx, ent_type] = num_existing_mention + 1 

                num_existing_entity = batch_passage_entity_length[batch_idx, passage_idx]
                if num_existing_entity < max_num_entity_per_passage:
                    batch_passage_entity_ids[batch_idx, passage_idx, num_existing_entity] = ent_type
                    batch_passage_entity_mask[batch_idx, passage_idx, num_existing_entity] = True
                    batch_passage_entity_length[batch_idx, passage_idx] = num_existing_entity + 1 

                batch_has_mention_entities[batch_idx].add(ent_type)

        for batch_idx, triples in enumerate(batch_triples):
            for head, rel, tail in triples:
                if not self.is_valid_triple(head, rel, tail, batch_max_num_entities, batch_has_mention_entities[batch_idx]):
                    continue
                existing_num_neighbor = batch_entity_num_edge[batch_idx, head]
                if existing_num_neighbor >= max_num_edge_per_entity:
                    continue
                existing_neighbors = set(batch_entity_adj[batch_idx, head, :existing_num_neighbor].tolist())
                if tail in existing_neighbors:
                    continue
                batch_entity_adj[batch_idx, head, existing_num_neighbor] = tail
                batch_entity_adj_mask[batch_idx, head, existing_num_neighbor] = True
                batch_entity_adj_relation[batch_idx, head, existing_num_neighbor] = self.relation2id[rel]
                batch_entity_num_edge[batch_idx, head] = existing_num_neighbor + 1 

        for batch_idx, triples in enumerate(batch_relevant_triples):
            for head, rel, tail in triples:
                existing_num_neighbor = batch_entity_num_edge[batch_idx, head]
                tail_index = (batch_entity_adj[batch_idx, head, :existing_num_neighbor] == tail).nonzero()
                if len(tail_index) == 0:
                    continue
                tail_index = tail_index[0].item()
                batch_entity_adj_relevant_relation_label[batch_idx, head, tail_index] = 1 

        max_question_len = (question_mask != 0).max(0)[0].nonzero(as_tuple=False)[-1].item() + 1 
        batch_question_indices = torch.arange(max_question_len)[None, :].expand(B, -1)
        batch_question_mask = batch_question_indices < question_length


        batch_entity_mention_indices = batch_entity_mention_indices + maxlen * batch_entity_mention_passage_indices
        batch_max_num_mention_per_entity = (batch_entity_mention_mask.reshape(-1, self.max_num_mention_per_entity) != 0).max(0)[0].nonzero(as_tuple=False)[-1].item() + 1 
        batch_entity_mention_indices = batch_entity_mention_indices[..., :batch_max_num_mention_per_entity]
        batch_entity_mention_mask = batch_entity_mention_mask[..., :batch_max_num_mention_per_entity]

        batch_max_num_entity_per_passage = (batch_passage_entity_mask.reshape(-1, max_num_entity_per_passage) != 0).max(0)[0].nonzero(as_tuple=False)[-1].item() + 1 
        batch_passage_entity_ids = batch_passage_entity_ids[..., :batch_max_num_entity_per_passage]
        batch_passage_entity_mask = batch_passage_entity_mask[..., :batch_max_num_entity_per_passage]

        batch_entity_text = [example["entity"] for example in batch]

        batch_entity_is_answer_label = self.get_entity_is_answer_label(batch, batch_max_num_entities)

        return (index, target_ids, target_mask, passage_ids, passage_masks, batch_question_text, batch_question_indices, batch_question_mask, \
                batch_entity_mention_indices, batch_entity_mention_mask, batch_entity_is_answer_label, batch_entity_text, \
                    batch_entity_adj, batch_entity_adj_mask, batch_entity_adj_relation, batch_entity_adj_relevant_relation_label, \
                        batch_passage_entity_ids, batch_passage_entity_mask)
    
    def maybe_truncate_question(self, question, max_num_words=100):
        words = question.split()
        if len(words) > max_num_words:
            question = " ".join(words[:max_num_words])
        return question 

    def is_valid_triple(self, head, rel, tail, max_num_entities, has_mention_entities):
        if head >= max_num_entities or tail >= max_num_entities:
            return False
        if rel not in self.relation2id:
            return False
        if head not in has_mention_entities or tail not in has_mention_entities:
            return False
        return True

    def get_entity_is_answer_label(self, batch, max_num_entities):

        batch_size = len(batch)
        entity_is_answer_label = torch.zeros((batch_size, max_num_entities), dtype=torch.long)
        for i, example in enumerate(batch):
            entity_is_answer_list = example["entity_is_answer_list"]
            num_entities = len(entity_is_answer_list)
            entity_is_answer_label[i, :num_entities] = torch.tensor(entity_is_answer_list, dtype=torch.long)

        return entity_is_answer_label

