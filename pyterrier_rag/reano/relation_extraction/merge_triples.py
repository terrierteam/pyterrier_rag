import os
import re 
import pickle 
import argparse
import numpy as np 
import networkx as nx 
from copy import deepcopy
from fuzzywuzzy import fuzz
from tqdm import tqdm, trange
from src.evaluation import ems, normalize_answer 

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data_folder", type=str, required=True, help="the common data folder of *_with_triples and *_with_pred_triples.pkl")
args = parser.parse_args()


def add_title_text_entity_triples(ctxs, triples):

    existing_hts = set()
    for head, rel, tail in triples:
        existing_hts.add((head, tail))
    
    new_triples = []
    for ctx in ctxs:
        entities = [ctx["title_entity"][0][3]] + [entity[3] for entity in ctx["text_entity"]]
        for j, e in enumerate(entities):
            if j == 0:
                # 表示的是title entity 
                candidate_neighbor_entity = [candidate for candidate in entities[1:] if candidate != e]
            else:
                candidate_neighbor_entity = [entities[0]] if entities[0] != e else []

            for ce in candidate_neighbor_entity:
                if (e, ce) in existing_hts:
                    continue 
                if (e, "UNK", ce) not in new_triples:
                    new_triples.append((e, "UNK", ce))
    
    return triples + new_triples

def fuzzy_match(texta, textb):
    if fuzz.token_sort_ratio(texta, textb) >= 80:
        return True
    return False

def has_numbers(input_string):
    return bool(re.search(r'[IVXLCDM\d]+', input_string))

def remove_bracket(text):

    text = text.strip()
    if "(" in text and ")" in text:
        left = text.find("(")
        right = text.find(")")
        if right > left and (not (left==0 and right==len(text)-1)):
            tmp_text = text[:left].strip()
            if right < len(text) - 1:
                tmp_text = tmp_text + " " + text[right+1:].strip()
            text = tmp_text.strip()
    text = text.replace("(", "").replace(")", "")
    return text

def get_entity_mention_name(item):
    
    entityid2mention_name = {}
    for ctx in item["ctxs"]:
        title_entityid = ctx["title_entity"][0][3]
        if title_entityid not in entityid2mention_name:
            entityid2mention_name[title_entityid] = remove_bracket(ctx["title_entity"][0][2])
        for start_idx, end_idx, mention, entityid in ctx["text_entity"]:
            if entityid not in entityid2mention_name:
                entityid2mention_name[entityid] = remove_bracket(mention)
    
    for entity_id, names in item["entityid2name"].items():
        if entity_id not in entityid2mention_name:
            entityid2mention_name[entity_id] = names[0]

    return entityid2mention_name


def get_entity_name(item):
    
    entityid2name = {}
    for entity_id, names in item["entityid2name"].items():
        if entity_id not in entityid2name:
            entityid2name[entity_id] = remove_bracket(names[0])
    return entityid2name


def fuzzy_match(text, answers):

    if any([has_numbers(ans) for ans in answers]):
        return False

    fuzzy_match = [fuzz.token_sort_ratio(normalize_answer(text), normalize_answer(ans)) >= 90 for ans in answers]
    if any(fuzzy_match):
        return True
    
    return False 


def get_relevant_triples(item, use_entity_name):

    question_entity = item["question_entity"]
    answers = item["answers"]
    if use_entity_name:
        entityid2name = get_entity_name(item)
    else:
        entityid2name = get_entity_mention_name(item) 
    entity_names = [entityid2name[i] for i in range(len(entityid2name))]
    entity_is_answer_list = [ems(entity_name, answers) for entity_name in entity_names]
    if np.sum(entity_is_answer_list) == 0:
        entity_is_answer_list = [fuzzy_match(entity_name, answers) for entity_name in entity_names]
    
    if not any(entity_is_answer_list):
        return [] 
    
    graph = nx.DiGraph()

    for i in range(len(entityid2name)):
        graph.add_node(i)

    ht2rel = {}
    for h, r, t in item["triples"]:
        ht2rel[(h, t)] = r 
        if not graph.has_edge(h, t):
            graph.add_edge(h, t)
    
    relevant_triples = [] 
    for question_entity_id in range(len(question_entity)):
        for ans_entity_id in [i for i in range(len(entityid2name)) if entity_is_answer_list[i]]:
            if question_entity_id == ans_entity_id:
                continue 
            try:
                paths = nx.shortest_path(graph, question_entity_id, ans_entity_id)
            except:
                paths = [] 

            if paths and len(paths) <= 4:
                for s, e in zip(paths[:-1], paths[1:]):
                    if (s, e) in ht2rel and (s, ht2rel[(s, e)], e) not in relevant_triples:
                        relevant_triples.append((s, ht2rel[(s, e)], e))
                    elif (e, s) in ht2rel and (e, ht2rel[(e, s)], s) not in relevant_triples:
                        relevant_triples.append((e, ht2rel[(e, s)], s))

    return relevant_triples 

        
def merge_triples(data_folder):

    use_entity_name = True if "entityquestion" in data_folder.lower() else False

    for file in ["test", "dev", "train"]:

        data_with_wikidata_triples = pickle.load(open(os.path.join(data_folder, f"{file}_with_triples.pkl"), "rb"))
        data_with_docunet_pred_triples = pickle.load(open(os.path.join(data_folder, f"{file}_with_pred_triples.pkl"), "rb"))
        data = deepcopy(data_with_wikidata_triples)

        assert len(data_with_wikidata_triples) == len(data_with_docunet_pred_triples)

        num_item_with_relevant_triples = 0 
        progress_bar = trange(len(data), desc="merging triples")
        for item, item1, item2 in zip(data, data_with_wikidata_triples, data_with_docunet_pred_triples):
            wikidata_triples = item1["triples"]
            docunet_triples = item2["docunet_pred_triples"]
            triples = list(set(wikidata_triples + docunet_triples))
            item["triples"] = triples
            item["relevant_triples"] = get_relevant_triples(item, use_entity_name)
            if len(item["relevant_triples"]) > 0:
                num_item_with_relevant_triples += 1 
            progress_bar.update(1)
                
        print("Proportion of data with relevant triples: {:4f} ({} / {})".format(num_item_with_relevant_triples / len(data), num_item_with_relevant_triples, len(data)))

        pickle.dump(data, open(os.path.join(data_folder, f"{file}_with_relevant_triples_wounkrel.pkl"), "wb"))


if __name__ == "__main__":
    
    merge_triples(args.data_folder)