import os
import sys
import spacy
import pickle
import argparse
import subprocess
import numpy as np 
from tqdm import tqdm 

import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from relation_extraction.load_data import Collator
from relation_extraction.docunet import DocREModel

# setup hyperparameters 
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# model parameters 
parser.add_argument("--unet_out_dim", type=int, default=256)
parser.add_argument("--unet_in_dim", type=int, default=3)
parser.add_argument("--down_dim", type=int, default=256, help="down_dim.")
parser.add_argument("--channel_type", type=str, default='context-based',help="unet_out_dim.")
parser.add_argument("--max_height", type=int, default=42,help="log.")

parser.add_argument("--relation2id", type=str, default=None, required=False, help="the file of relation name to relation id file.")
parser.add_argument("--docunet_checkpoint", type=str, default=None, required=False, help="the file of the DocuNet checkpoint")
parser.add_argument("--data_folder", type=str, default=None, required=False, help="the data folder of *_with_triples.pkl")

args = parser.parse_args()

model_name_or_path = "bert-base-uncased"
max_seq_length = 256 
checkpoint_path = args.docunet_checkpoint 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

global model, nlp, tokenizer, collator, id2relation
model=None

def load_model(checkpoint_path, relation2id_path):

    global nlp, tokenizer, collator, id2relation, model

    print("Loading Spacy Model ... ")
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")

    print("loading tokenizer ... ")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # collator = Collator(tokenizer, relation2id=args.relation2id, max_length=max_seq_length)
    collator = Collator(tokenizer, relation2id=relation2id_path, max_length=max_seq_length)
    id2relation = {v: k for k, v in collator.relation2id.items()}

    print("loading docunet model ...")
    config = AutoConfig.from_pretrained(model_name_or_path, num_labels=collator.num_relation)
    model = AutoModel.from_pretrained(model_name_or_path, config=config)
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = "bert"
    model = DocREModel(config, args, model, num_labels=10)
    # print(f"loading checkpoint from {checkpoint_path} ...")
    # ckpt = torch.load(checkpoint_path, map_location="cpu")
    print(f"loading checkpoint from {checkpoint_path} ...")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    print("Finish Initializing!")


if args.relation2id is not None and args.docunet_checkpoint is not None:
    load_model(args.docunet_checkpoint, args.relation2id)


def spacy_entity_type(text):
    entity_type = "PROPN"
    for ent in nlp(text).ents:
        entity_type = ent.label_
    return entity_type


def remove_entity_symbol(text):
    return text.replace("<e> ", "").replace(" </e>", "")


def convert_qa_data_to_rebel_format(ctx, entity2type=None):

    new_title = remove_entity_symbol(ctx["title"]).strip()
    entity2id, entitypos, entitytype = {}, [], []
    new_text = "title: {}, text: ".format(new_title)
    prefix_index = len(new_text)
    new_text += remove_entity_symbol(ctx["text"])

    if "title_entity" in ctx and len(ctx["title_entity"])>0:
        entity2id[ctx["title_entity"][0][3]] = len(entity2id)
        entitypos.append([])
        entitypos[-1].append((len("title: "), len("title: ")+len(new_title)))
        entitytype.append("PROPN" if entity2type is None else entity2type[ctx["title_entity"][0][3]])

    for i, (start_idx, end_idx, mention, entity) in enumerate(ctx["text_entity"]):

        new_start_idx = prefix_index + (start_idx-i*9-4)
        new_end_idx = prefix_index + (end_idx-i*9-4)

        assert new_end_idx - new_start_idx == len(mention)
        if new_text[new_start_idx: new_end_idx] != mention:
            print("Original Text: {}\nOriginal Span: ({}, {}, {})\nNew Text: {}\nNew Span: ({}, {}, {})".format(\
                ctx["text"], start_idx, end_idx, mention, new_text, new_start_idx, new_end_idx, \
                    new_text[new_start_idx: new_end_idx]))
        
        if entity in entity2id:
            entitypos[entity2id[entity]].append((new_start_idx, new_end_idx))
        else:
            entity2id[entity] = len(entity2id)
            entitypos.append([])
            entitypos[-1].append((new_start_idx, new_end_idx))
            entitytype.append("PROPN" if entity2type is None else entity2type[entity])

    output = {
        "title": new_title,
        "text": new_text, 
        "entity": [item[0] for item in sorted(entity2id.items(), key=lambda x: x[1])], 
        "entitypos": entitypos,
        "entitytype": entitytype, 
        "triples": [],
    }

    return output


def relation_extraction_for_one_question(item, checkpoint_path=None, relation2id_path=None, batch_size=10):

    global model, nlp, tokenizer, collator, id2relation

    if model is None and (checkpoint_path is not None and relation2id_path is not None):
        load_model(checkpoint_path, relation2id_path)

    entity2type = {}
    for entity in item["entityid2name"]:
        entity_name = item["entityid2name"][entity][0]
        entity_type = spacy_entity_type(entity_name)
        entity2type[entity] = entity_type
    
    new_ctxs = []
    for ctx in item["ctxs"]:
        new_ctxs.append(convert_qa_data_to_rebel_format(ctx=ctx, entity2type=entity2type))
    
    pred_triples_list = [] 

    # batch_size = 10
    for i in range((len(new_ctxs) - 1)//batch_size + 1):

        batch_ctxs = new_ctxs[i*batch_size: (i+1)*batch_size]
        batch = collator(batch_ctxs)
        mask = [len(hts) > 0 for hts in batch[4]]
        input_features = {
            "input_ids": batch[0][mask].to(device),
            "attention_mask": batch[1][mask].to(device),
            "entity_pos": [entity_pos for m, entity_pos in zip(mask, batch[3]) if m], 
            "hts": [hts for m, hts in zip(mask, batch[4]) if m],
        }
        pred_relations, pred_logits = model(**input_features)
        pred_relations = pred_relations.detach().cpu().numpy()
        pred_logits = pred_logits.detach().cpu().numpy()

        assert len(sum(input_features["hts"], [])) == len(pred_relations)

        prev_hts_num = 0 
        for hts, entity2id in zip(batch[4], batch[5]):

            if len(hts) == 0:
                continue

            hts_num = len(hts)
            idx2entity = {v:k for k, v in entity2id.items()}
            triples_list = []

            for (h, t), pred_relations_vector, pred_relations_logits in zip(
                hts, 
                pred_relations[prev_hts_num: prev_hts_num+hts_num],
                pred_logits[prev_hts_num: prev_hts_num+hts_num]
            ):
                pred_rel_indices = pred_relations_vector.nonzero()[0]

                if len(pred_rel_indices) == 1 and pred_rel_indices[0] == 0:
                    if h == 0 or t == 0:
                        pred_rel_idx = np.argmax(pred_relations_logits)
                        if pred_rel_idx != 0:
                            triples_list.append((idx2entity[h], id2relation[pred_rel_idx], idx2entity[t]))

                for pred_rel_idx in pred_rel_indices:
                    if pred_rel_idx == 0:
                        continue
                    triples_list.append((idx2entity[h], id2relation[pred_rel_idx], idx2entity[t]))
            prev_hts_num += hts_num
            pred_triples_list.append(triples_list)

    return_item = {}
    return_item["docunet_pred_triples"] = sum(pred_triples_list, [])

    return return_item


def relation_extraction(file, save_file):

    from tqdm import tqdm 
    
    data = pickle.load(open(file, "rb"))
    new_data = []
    for item in tqdm(data):
        new_data.append(relation_extraction_for_one_question(item))
    pickle.dump(new_data, open(save_file, "wb"))


if __name__ == "__main__":

    data_folder = args.data_folder 
    # relation_extraction(os.path.join(data_folder, "train_with_triples.pkl"), os.path.join(data_folder, "train_with_pred_triples.pkl"))
    relation_extraction(os.path.join(data_folder, "dev_with_triples.pkl"), os.path.join(data_folder, "dev_with_pred_triples.pkl"))
    # relation_extraction(os.path.join(data_folder, "test_with_triples.pkl"), os.path.join(data_folder, "test_with_pred_triples.pkl"))