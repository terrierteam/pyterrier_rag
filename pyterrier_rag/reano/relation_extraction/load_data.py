import pickle
import logging
from tqdm import tqdm 
from collections import defaultdict

import torch
from torch.utils.data import Dataset

from src.util import load_json
logger = logging.getLogger(__name__)


class REBEL(Dataset):

    def __init__(self, file, entity2type: str=None):

        print(f"loading data from {file} ...")
        self.data = load_json(file, type="jsonl")
        print(f"loading entity2type from {entity2type} ...")
        self.entity2type = pickle.load(open(entity2type, "rb")) if entity2type is not None else None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):

        # {
        #     'docid': "30113940", 
        #     'title': 'Transportation Journal', 
        #     'uri': 'Q7835172', 
        #     'text': 'xxx', 
        #     'entities': [
        #         {'uri': 'Q737498', 'boundaries': [29, 45], 'surfaceform': 'academic journal', 'annotator': 'Me'}, 
        #         {'uri': 'Q7163300', 'boundaries': [145, 172], 'surfaceform': 'Penn State University Press', 'annotator': 'Me'},
        #     ] 
        #     'triples':[
        #         {
        #             'subject': {'uri': 'Q7835172', 'boundaries': [0, 22], 'surfaceform': 'Transportation Journal', 'annotator': 'Me'}, 
        #             'predicate': {'uri': 'P123', 'boundaries': None, 'surfaceform': 'publisher', 'annotator': 'NoSubject-Triple-aligner'}, 
        #             'object': {'uri': 'Q7163300', 'boundaries': [145, 172], 'surfaceform': 'Penn State University Press', 'annotator': 'Me'}, 
        #             'sentence_id': 1, 'dependency_path': None, 'confidence': None, 'annotator': 'NoSubject-Triple-aligner'
        #          }
        #     ]
        # }

        example = self.data[index]
        entity2id = {}
        entitypos = []
        entitytype = []

        text = "title: {}, text: ".format(example["title"])
        prefix_index = len(text)
        text += example["text"]

        entity2id[example["uri"]] = len(entity2id)
        entitypos.append([])
        entitypos[-1].append((len("title: "), len("title: ") + len(example["title"])))
        entitytype.append("PROPN" if self.entity2type is None else self.entity2type[example["uri"]])

        for entity in example["entities"]:
            eid = entity["uri"]
            if eid in entity2id:
                entitypos[entity2id[eid]].append((prefix_index+entity["boundaries"][0], prefix_index+entity["boundaries"][1]))
            else:
                entity2id[eid] = len(entity2id)
                entitypos.append([])
                entitypos[-1].append((prefix_index+entity["boundaries"][0], prefix_index+entity["boundaries"][1]))
                entitytype.append("PROPN" if self.entity2type is None else self.entity2type[entity["uri"]])
        
        triples = []
        for triple in example["triples"]:
            heid = triple["subject"]["uri"]
            teid = triple["object"]["uri"]
            if heid in entity2id and teid in entity2id:
                triples.append((heid, triple["predicate"]["uri"], teid))
        
        assert len(entity2id) == len(entitypos)
        assert len(entity2id) == len(entitytype)

        return {
            "title": example["title"],
            "text": text, 
            "entity": [item[0] for item in sorted(entity2id.items(), key=lambda x: x[1])],
            "entitypos": entitypos,
            "entitytype": entitytype,
            "triples": triples
        }


class Collator:

    def __init__(self, tokenizer, relation2id, max_length=256, max_num_entities=25):

        self.tokenizer = tokenizer
        self.tokenizer.add_special_tokens({"additional_special_tokens":[f"[unused{i}]" for i in range(100)]})
        self.max_length = max_length
        self.max_num_entities = max_num_entities
        self.entity_types = ["PROPN", 'CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', \
                             'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART']
        self.entitytype2id = {t:i for i, t in enumerate(self.entity_types)}
        self.relation2id = pickle.load(open(relation2id, "rb"))
        self.unknown_relation = "UNK"
        self.num_relation = len(self.relation2id)

    
    def __call__(self, batch):

        features = [self.convert_example_to_features(example) for example in batch]
        max_len = min(max([len(feature["input_ids"]) for feature in features]), self.max_length)
        input_ids = [feature["input_ids"] + [0] * (max_len-len(feature["input_ids"])) for feature in features]
        input_mask = [[1] * len(feature["input_ids"]) + [0] * (max_len-len(feature["input_ids"])) for feature in features]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        entity_pos = [feature["entity_pos"] for feature in features]
        labels = [feature["labels"] for feature in features]
        hts = [feature["hts"] for feature in features]
        entity2id_list = [feature["entity2id"] for feature in features]

        output = (input_ids, input_mask, labels, entity_pos, hts, entity2id_list)

        return output
    
    def sort_spans(self, spans):
        sorted_spans = sorted(spans, key=lambda x: (x[0], x[0]-x[1]))
        filter_spans = []
        for span in sorted_spans:
            if len(filter_spans) == 0:
                filter_spans.append(span)
            else:
                if span[0] >= filter_spans[-1][1]:
                    filter_spans.append(span)
        return filter_spans[:self.max_num_entities]

    
    def convert_example_to_features(self, example):
        
        # {
        #     "title": title,
        #     "text": text, 
        #     "entity": ["Qxxx"], 
        #     "entitypos": [[(), ()], ...],
        #     "entitytype": ["PROPN"],
        #     "triples": [("Qxxx", "Pxxx", "Qxxx")]
        # }

        input_tokens = [] 
        entity_token_spans = {}
        num_special_tokens = 2 

        all_entity_spans = []
        for entity, spans, entity_type in zip(example["entity"], example["entitypos"], example["entitytype"]):
            for span in spans:
                all_entity_spans.append((span[0], span[1], entity, entity_type))
        all_entity_spans = self.sort_spans(all_entity_spans)

        if len(all_entity_spans) == 0:
            # print(example)
            input_tokens = self.tokenizer.tokenize(example["text"])[:self.max_length-num_special_tokens]
            input_tokens = ["[CLS]"] + input_tokens + ["[SEP]"]
            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
            return {
                "input_ids": input_ids,
                'entity_pos': [],
                "labels": [],
                'hts': [],
                'title': example['title'],
                'entity2id': {},
            }
        
        text = example["text"]
        input_tokens.extend(self.tokenizer.tokenize(text[:all_entity_spans[0][0]]))
        for i, (start_idx, end_idx, entity, entity_type) in enumerate(all_entity_spans):
            entity_type_id = self.entitytype2id[entity_type]
            entity_text = "[unused{}] {} [unused{}]".format(entity_type_id, text[start_idx: end_idx], entity_type_id+50)
            entity_tokens = self.tokenizer.tokenize(entity_text)
            if len(input_tokens) + len(entity_tokens) + num_special_tokens > self.max_length:
                break
            if entity not in entity_token_spans:
                entity_token_spans[entity] = []
            entity_token_spans[entity].append((len(input_tokens), len(input_tokens)+len(entity_tokens)))
            input_tokens.extend(entity_tokens)

            if i < len(all_entity_spans) - 1:
                input_tokens.extend(self.tokenizer.tokenize(text[end_idx: all_entity_spans[i+1][0]]))
            if len(input_tokens) + len(entity_tokens) + num_special_tokens > self.max_length:
                break
        input_tokens.extend(self.tokenizer.tokenize(text[all_entity_spans[-1][1]:]))
        input_tokens = input_tokens[:self.max_length-num_special_tokens]

        input_ids = self.tokenizer.convert_tokens_to_ids(["[CLS]"] + input_tokens + ["[SEP]"])

        entities = list(entity_token_spans.keys())
        entity2id = {entity:i for i, entity in enumerate(entities)}

        entity_pos = []
        for entity in entities:
            entity_pos.append([])
            for start_token_idx, end_token_idx in entity_token_spans[entity]:
                entity_pos[-1].append((start_token_idx, end_token_idx))
                
        entity_pair2relations = defaultdict(list)
        for heid, relation, teid in example["triples"]:
            if heid not in entities or teid not in entities:
                continue
            entity_pair2relations[(heid, teid)].append(relation)

        relations, hts = [], [] 
        for heid, teid in entity_pair2relations.keys():
            relation = [0] * len(self.relation2id)
            for r in entity_pair2relations[(heid, teid)]:
                if r not in self.relation2id:
                    r = self.unknown_relation
                relation[self.relation2id[r]] = 1 
            hts.append((entity2id[heid], entity2id[teid]))
            relations.append(relation)

        for heid in entities:
            for teid in entities:
                if heid != teid and (entity2id[heid], entity2id[teid]) not in hts:
                    relation = [0] * len(self.relation2id)
                    relation[self.relation2id[self.unknown_relation]] = 1 
                    hts.append((entity2id[heid], entity2id[teid]))
                    relations.append(relation)

        assert len(relations) == len(entities) * (len(entities) - 1)

        return {
            "input_ids": input_ids,
            'entity_pos': entity_pos,
            "labels": relations,
            'hts': hts,
            'title': example['title'],
            "entity2id": entity2id
        }