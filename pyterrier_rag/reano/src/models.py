import os 
os.environ["JAVA_HOME"] = "/nfs/java/jdk-11.0.1"
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
import pyterrier as pt 
if not pt.started():
    pt.init()

import os
import torch 
import pickle
from pandas import DataFrame
from tqdm import trange

from transformers import T5Tokenizer
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

from src.fid import TripleKGFiDT5
from src.datasets import FiDCollator
from src.util import remove_bracket
from src.entity_recognition import ner_spacy_tagme, get_triples_from_wikidata
# import relation_extraction.docunet_inference
from relation_extraction.docunet_inference import relation_extraction_for_one_question

class REANO(pt.Transformer):

    # def __init__(self, model_path: str, relation_path: str, text_maxlength: int=250, answer_maxlength: int=25, num_neighbors: int=20, batch_size=4):
    def __init__(self, model_name_or_path: str, docunet_model_name_or_path: str, text_maxlength: int=250, answer_maxlength: int=25, num_neighbors: int=20, batch_size=4):

        super().__init__()

        self.device = torch.device("cuda")
        self.batch_size = batch_size
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength
        self.num_neighbors = num_neighbors

        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.tokenizer.add_special_tokens({'sep_token': '[SEP]', 'cls_token': '[CLS]'})
        self.tokenizer.add_tokens(["<e>", "</e>"])

        model_folder = self.get_model_folder(model_name_or_path)
        docunet_model_folder = self.get_model_folder(docunet_model_name_or_path)

        relation2id = pickle.load(open(os.path.join(docunet_model_folder, "relation2id.pkl"), "rb"))
        relationid2name = pickle.load(open(os.path.join(docunet_model_folder, "relationid2name.pkl"), "rb"))
        relation_embedding = pickle.load(open(os.path.join(docunet_model_folder, "relation_t5base_embeddings.pkl"), "rb"))

        self.collator = Collator(
            text_maxlength=text_maxlength, 
            tokenizer=self.tokenizer,
            relation2id=relation2id, 
            answer_maxlength=answer_maxlength, 
            max_num_edge=self.num_neighbors, # NOTE: this value should be the same as the k value in the TripleKGFiDT5 model 
        )

        self.model = self.initialize_model(model_folder, relationid2name, relation_embedding)
    
    def get_model_folder(self, model_name_or_path: str):

        path_map = {
            "Jinyuan6/docunet": os.path.expanduser("~/.cache/huggingface/transformers/docunet"), 
            "Jinyuan6/reano_2wiki": os.path.expanduser("~/.cache/huggingface/transformers/reano_2wiki"), 
            "Jinyuan6/reano_nq": os.path.expanduser("~/.cache/huggingface/transformers/reano_nq"), 
            "Jinyuan6/reano_tqa": os.path.expanduser("~/.cache/huggingface/transformers/reano_tqa"), 
            "Jinyuan6/reano_musique": os.path.expanduser("~/.cache/huggingface/transformers/reano_musique"), 
            "Jinyuan6/reano_eq": os.path.expanduser("~/.cache/huggingface/transformers/reano_eq"), 
        }

        if model_name_or_path in path_map.keys():
            model_path = path_map[model_name_or_path]
            if not os.path.exists(model_path):
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=model_name_or_path, local_dir=model_path, force_download=True, local_dir_use_symlinks=False)
        else:
            model_path = model_name_or_path
            if not os.path.exists(model_path):
                raise ValueError(f"folder {model_path} does not exists!")
        
        return model_path
    
    def initialize_model(self, model_path, relationid2name, relation_embedding):
        model = TripleKGFiDT5.from_pretrained(
            "t5-base", tokenizer=self.tokenizer, ent_dim=128, k=self.num_neighbors, hop=3,
        )
        model.resize_token_embeddings(len(self.tokenizer))
        model.relation_extraction_setup(
            relationid2name=relationid2name, 
            relation_embedding=relation_embedding
        )

        # load model checkpoint 
        print(f"loading model checkpoing from {model_path} ...")
        model.load_state_dict(torch.load(os.path.join(model_path, "checkpoint.ckpt"), map_location="cpu")["model"])

        model.eval()
        model.to(self.device)
        return model 
    
    def transform(self, run):
        """
        run: 
        qid | question | ctxs | question_entity | entityid2name | triples
        """
        inputs = self.get_collator_inputs(run)
        pred_answers = [] 
        for i in trange((len(run)-1) // self.batch_size + 1, desc="generating answers"):
            batch_inputs = self.collator(inputs[i*self.batch_size: (i+1)*self.batch_size])
            batch_pred_answers = self.pred_answer(batch_inputs)
            pred_answers.extend(batch_pred_answers)
        run["answer"] = pred_answers
        return run
    
    def pred_answer(self, inputs):
        inputs = [data.to(self.device) if torch.is_tensor(data) else data for data in inputs]
        qids, passage_ids, passage_mask, question_text, question_indices, question_mask, \
            ent_indices, ent_mask, entity_text, entity_adj, entity_adj_mask, entity_adj_relation = inputs
        outputs = self.model.generate(
            input_ids=passage_ids,
            attention_mask=passage_mask,
            question_indices=question_indices,
            question_mask=question_mask,
            ent_indices=ent_indices,
            ent_mask=ent_mask,
            entity_text=entity_text, 
            entity_adj=entity_adj, 
            entity_adj_mask=entity_adj_mask, 
            entity_adj_relation=entity_adj_relation, 
            max_length=self.answer_maxlength,
            question_text=question_text,
        )
        preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return preds 
    
    def get_collator_inputs(self, run: DataFrame):

        inputs = [] 
        n_context = max([len(run.iloc[i]["ctxs"]) for i in range(len(run))])

        for example in run.to_dict(orient='records'):
            question = "question: " + example["question"]
            entityid2name = example["entityid2name"]
            batch_ent_map = {}
            batch_entity_name = []

            question_entity_list = [] 
            for i, question_entity in enumerate(example["question_entity"]):
                batch_ent_map[i] = len(batch_ent_map)
                entity_name = remove_bracket(question_entity)
                batch_entity_name.append(entity_name)
                question_entity_list.append(batch_ent_map[i])
            
            f = "title: {title} context: {text}"
            passages, entity_type_list = [], [] 
            for c in example["ctxs"]:
                title, title_entity = self.get_context_text(c["title"], c["title_entity"], batch_ent_map, batch_entity_name, entityid2name)
                text, text_entity = self.get_context_text(c["text"], c["text_entity"], batch_ent_map, batch_entity_name, entityid2name)
                passages.append(f.format(title=title, text=text))
                entity_type_list.append(title_entity+text_entity)
            
            if len(passages) < n_context:
                while len(passages) < n_context:
                    passages = passages + passages[:(n_context-len(passages))]
                    entity_type_list = entity_type_list + entity_type_list[:(n_context-len(entity_type_list))]

            num_entity = len(batch_ent_map)
            triples = [] 
            for heid, rel, teid in example.get("triples", []): 
                if heid in batch_ent_map and teid in batch_ent_map:
                    triple = (batch_ent_map[heid], rel, batch_ent_map[teid])
                    if triple not in triples:
                        triples.append(triple)
            
            inputs.append(
                {
                    "qid": example["qid"], 
                    "question": question, 
                    "passages": passages, 
                    "num_entity": num_entity, 
                    "entity": batch_entity_name, 
                    "entity_type_list": entity_type_list, 
                    "triples": triples, 
                }
            )
        
        return inputs
    
    def get_context_text(self, text_with_entity, entity_list, batch_ent_map, batch_entity_name, entityid2name):

        entity_type = []
        sorted_entity_list = sorted(entity_list, key=lambda x: x[0])
        for start_idx, end_idx, mention, entity_id in sorted_entity_list:

            if entity_id not in batch_ent_map:
                batch_ent_map[entity_id] = len(batch_ent_map)
                # if not self.use_entityname:
                entity_name = remove_bracket(mention)
                # else:
                #     entity_name = remove_bracket(entityid2name[entity_id][0])
                batch_entity_name.append(entity_name)
            
            entity_type.append(batch_ent_map[entity_id])

        return text_with_entity, entity_type

class EntityRecognition(pt.Transformer):

    def transform(self, run: DataFrame) -> DataFrame:
        """
        Input: 
        run: qid | question | ctxs
        Output:
        run: qid | question | ctxs | question_entity | entityid2name
        """
        if "question" not in run.columns and "query" in run.columns:
            run["question"] = run["query"]
        if "ctxs" not in run.columns and "text" in run.columns:
            new_run_dict_data = []
            for qid in run["qid"].unique():
                qid_run = run[run["qid"] == qid]
                one_dict_data = {
                    'qid': qid, 
                    "query": qid_run.iloc[0]["query"], 
                    "question": qid_run.iloc[0]["question"], 
                }
                ctxs = []
                for i in range(len(qid_run)):
                    one_ctx = {"id": str(qid_run.iloc[i]["docno"])}
                    if "title" in qid_run.columns:
                        one_ctx["title"] = qid_run.iloc[i]["title"]
                    else:
                        one_ctx["title"] = "" 
                    one_ctx["text"] = qid_run.iloc[i]["text"]
                    ctxs.append(one_ctx)
                one_dict_data["ctxs"] = ctxs
                if "gold_answer" in qid_run.columns:
                    one_dict_data["gold_answer"] = qid_run.iloc[0]["gold_answer"]
                new_run_dict_data.append(one_dict_data)
            run = DataFrame.from_records(new_run_dict_data)
        
        new_ctxs_list, question_entities_list, entityid2name_list = [], [], []
        progress_bar = trange(len(run), desc="Identifying Entities")
        for example in run.to_dict(orient='records'):
            ner_results = ner_spacy_tagme(example)
            new_ctxs_list.append(ner_results["ctxs"])
            question_entities_list.append(ner_results["question_entity"])
            entityid2name_list.append(ner_results["entityid2name"])
            progress_bar.update(1)
        
        run["ctxs"] = new_ctxs_list
        run["question_entity"] = question_entities_list
        run["entityid2name"] = entityid2name_list

        return run
    
class WikidataRelationExtraction(pt.Transformer):

    def transform(self, run: DataFrame) -> DataFrame:
        """
        Input: 
        run: qid | question | ctxs | question_entity | entityid2name 
        Output:
        run: qid | question | ctxs | question_entity | entityid2name | entityid2wikidataid | wikidata_triples
        """
        entityid2wikidataid_list, triples_list = [], [] 
        progress_bar = trange(len(run), desc="Identifying Wikidata Relation")
        for example in run.to_dict(orient='records'):
            wikidata_results = get_triples_from_wikidata(example)
            entityid2wikidataid_list.append(wikidata_results["entityid2wikidataid"])
            triples_list.append(wikidata_results["triples"])
            progress_bar.update(1)
        
        run["entityid2wikidataid"] = entityid2wikidataid_list
        run["wikidata_triples"] = triples_list
    
        return run


class DocuNetRelationExtraction(pt.Transformer):

    # def __init__(self, model_path: str, relation_path: str, batch_size: int=10):
    def __init__(self, model_name_or_path: str, batch_size: int=10):

        super().__init__()
        self.device = torch.device("cuda")
        # self.model_path = model_path
        # self.relation2id_path = os.path.join(relation_path, "relation2id.pkl")
        self.model_folder = self.get_model_folder(model_name_or_path)
        self.model_path = os.path.join(self.model_folder, "docunet.ckpt")
        self.relation2id_path = os.path.join(self.model_folder, "relation2id.pkl")
        self.batch_size = batch_size
    
    def get_model_folder(self, model_name_or_path):

        if model_name_or_path == "Jinyuan6/docunet":
            model_path = os.path.expanduser("~/.cache/huggingface/transformers/docunet")
            if not os.path.exists(model_path):
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=model_name_or_path, local_dir=model_path, force_download=True, local_dir_use_symlinks=False)
        else:
            model_path = model_name_or_path
            if not os.path.exists(model_path):
                raise ValueError(f"folder {model_path} does not exists!")
        
        return model_path
    
    def transform(self, run: DataFrame) -> DataFrame:
        """
        Input:
        run: qid | question | ctxs | entityid2name
        Output:
        run: qid | question | ctxs | entityid2name | docunet_pred_triples
        """
        relations = []
        progress_bar = trange(len(run), desc="DocuNet Relation Extraction")
        for example in run.to_dict(orient='records'):
            results = relation_extraction_for_one_question(example, self.model_path, self.relation2id_path, self.batch_size)
            relations.append(results["docunet_pred_triples"])
            progress_bar.update(1)
        run["docunet_pred_triples"] = relations
        return run 

def encode_passages(batch_text_passages, tokenizer, max_length):

    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=max_length,
            # pad_to_max_length=True,
            padding="max_length",
            return_tensors='pt',
            truncation=True
        )
        passage_ids.append(p['input_ids'][None])
        passage_masks.append(p['attention_mask'][None])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)

    return passage_ids, passage_masks.bool()

class Collator(object):

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

        # qids = torch.tensor([ex['qid'] for ex in batch])
        qids = [ex["qid"] for ex in batch]  
        def append_question(example):
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

        # max_num_entity_per_passage = 25 
        # batch_passage_entity_length = torch.zeros((batch_size, num_passages), dtype=torch.long)
        # batch_passage_entity_ids = torch.zeros((batch_size, num_passages, max_num_entity_per_passage), dtype=torch.long)
        # batch_passage_entity_mask = torch.zeros((batch_size, num_passages, max_num_entity_per_passage), dtype=torch.bool)

        max_num_edge_per_entity = self.max_num_edge
        batch_entity_adj = torch.zeros((batch_size, batch_max_num_entities, max_num_edge_per_entity), dtype=torch.long)
        batch_entity_num_edge = torch.zeros((batch_size, batch_max_num_entities), dtype=torch.long)
        batch_entity_adj_mask = torch.zeros((batch_size, batch_max_num_entities, max_num_edge_per_entity), dtype=torch.bool)
        batch_entity_adj_relation = torch.zeros((batch_size, batch_max_num_entities, max_num_edge_per_entity), dtype=torch.long) 
        # batch_entity_adj_relevant_relation_label = torch.zeros((batch_size, batch_max_num_entities, max_num_edge_per_entity), dtype=torch.long) 
        batch_triples = [example["triples"] for example in batch]
        # batch_relevant_triples = [example["relevant_triples"] for example in batch] 
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

                # num_existing_entity = batch_passage_entity_length[batch_idx, passage_idx]
                # if num_existing_entity < max_num_entity_per_passage:
                #     batch_passage_entity_ids[batch_idx, passage_idx, num_existing_entity] = ent_type
                #     batch_passage_entity_mask[batch_idx, passage_idx, num_existing_entity] = True
                #     batch_passage_entity_length[batch_idx, passage_idx] = num_existing_entity + 1 

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

        # for batch_idx, triples in enumerate(batch_relevant_triples):
        #     for head, rel, tail in triples:
        #         existing_num_neighbor = batch_entity_num_edge[batch_idx, head]
        #         tail_index = (batch_entity_adj[batch_idx, head, :existing_num_neighbor] == tail).nonzero()
        #         if len(tail_index) == 0:
        #             continue
        #         tail_index = tail_index[0].item()
        #         batch_entity_adj_relevant_relation_label[batch_idx, head, tail_index] = 1 

        max_question_len = (question_mask != 0).max(0)[0].nonzero(as_tuple=False)[-1].item() + 1 
        batch_question_indices = torch.arange(max_question_len)[None, :].expand(B, -1)
        batch_question_mask = batch_question_indices < question_length

        batch_entity_mention_indices = batch_entity_mention_indices + maxlen * batch_entity_mention_passage_indices
        batch_max_num_mention_per_entity = (batch_entity_mention_mask.reshape(-1, self.max_num_mention_per_entity) != 0).max(0)[0].nonzero(as_tuple=False)[-1].item() + 1 
        batch_entity_mention_indices = batch_entity_mention_indices[..., :batch_max_num_mention_per_entity]
        batch_entity_mention_mask = batch_entity_mention_mask[..., :batch_max_num_mention_per_entity]

        # batch_max_num_entity_per_passage = (batch_passage_entity_mask.reshape(-1, max_num_entity_per_passage) != 0).max(0)[0].nonzero(as_tuple=False)[-1].item() + 1 
        # batch_passage_entity_ids = batch_passage_entity_ids[..., :batch_max_num_entity_per_passage]
        # batch_passage_entity_mask = batch_passage_entity_mask[..., :batch_max_num_entity_per_passage]

        batch_entity_text = [example["entity"] for example in batch]
        # batch_entity_is_answer_label = self.get_entity_is_answer_label(batch, batch_max_num_entities)

        return (qids, passage_ids, passage_masks, batch_question_text, batch_question_indices, batch_question_mask, \
                    batch_entity_mention_indices, batch_entity_mention_mask, batch_entity_text,
                        batch_entity_adj, batch_entity_adj_mask, batch_entity_adj_relation)

        # return (index, target_ids, target_mask, passage_ids, passage_masks, batch_question_text, batch_question_indices, batch_question_mask, \
        #         batch_entity_mention_indices, batch_entity_mention_mask, batch_entity_is_answer_label, batch_entity_text, \
        #             batch_entity_adj, batch_entity_adj_mask, batch_entity_adj_relation, batch_entity_adj_relevant_relation_label, \
        #                 batch_passage_entity_ids, batch_passage_entity_mask)
    
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


if __name__ == "__main__":

    import pickle 
    import ir_measures
    from ir_measures import EM, F1

    print("loading data ...")
    data = pickle.load(open("/nfs/common/data/2wikimultihopqa/reano_data/test_with_relevant_triples_wounkrel.pkl", "rb"))
    df_data = DataFrame.from_records(data=data)
    df_data["gold_answer"] = df_data["answers"]
    df_data["qid"] = list(range(len(df_data)))

    model = REANO(
        model_path="/nfs/reano/checkpoints/2wikimultihopqa", 
        relation_path="/nfs/reano/rebel_dataset", 
    )
    results = model.transform(df_data)
    
    metrics = ir_measures.calc_aggregate([EM, F1], qrels=df_data[["qid", "gold_answer"]], run=results)
    print(metrics)