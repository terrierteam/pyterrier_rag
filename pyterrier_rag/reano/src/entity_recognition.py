import re 
import sys
import time
import tagme 
tagme.GCUBE_TOKEN = "4d1002a7-8bae-4adc-97d5-8be425ffbf14-843339462"
import spacy
import string 
import subprocess
from collections import defaultdict

print("Loading Spacy Model ... ")
try:
    nlp = spacy.load("en_core_web_sm")
except:
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

def get_tagme_span(text, threshold=0.25):

    max_retries = 3
    retry_delay = 60

    for attempt in range(max_retries):
        try:
            spans = []
            for span in tagme.annotate(text).get_annotations(threshold):
                spans.append((span.begin, span.end, span.mention, span.entity_title))
            break
        except Exception as e:
            print(f"Error: {e}")
            # print(text)
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print(f"Max retries exceeded. Could not establish a connection. Give up Tagme for {text}!")

            spans = [] 

    return spans
            
def process_tagme_one_item(text, threshold):

    tagme_spans = get_tagme_span(text, threshold) #[(start, end, mention, entity_name)]
    
    sorted_tagme_spans = sorted(tagme_spans, key=lambda x: (x[0], x[0]-x[1]))
    filter_tagme_spans = []
    for item in sorted_tagme_spans:
        if len(filter_tagme_spans) == 0:
            filter_tagme_spans.append(item)
        else:
            prev_end = filter_tagme_spans[-1][1]
            if item[0] >= prev_end:
                filter_tagme_spans.append(item)
    
    prefix = 0 
    clean_tagme_spans = [] 
    for start_idx, end_idx, mention, entity in filter_tagme_spans:
        if text[start_idx: end_idx] == mention:
            new_start_idx, new_end_idx = start_idx, end_idx
        else:
            new_start_idx = text.find(mention, prefix)
            new_end_idx = new_start_idx + len(mention)
            if new_start_idx < 0:
                continue
        clean_tagme_spans.append((new_start_idx, new_end_idx, mention, entity))
        prefix = new_end_idx

    tagme_spans = clean_tagme_spans
    spans, mentions, entity_names = [], [], []
    for span in tagme_spans:
        spans.append((span[0], span[1]))
        mentions.append(span[2])
        entity_names.append(span[3])
        
    return {
        "text": text,
        "span": spans,
        "mention": mentions,
        "entity_name": entity_names
    }

def get_re_pattern(pattern):
    return pattern.replace("\\", "\\\\").replace("+", "\+").replace(".", "\.").replace("$", "\$").replace("?", "\?").replace("^", "\^").replace("*", "\*").replace("(", "\(").replace(")", "\)").replace("{", "\{").replace("}", "\}").replace("[", "\[").replace("]", "\]").replace("|", "\|")

def get_entities_spans(entities, text):

    span_list = [] 
    for i, entity in enumerate(entities):
        if len(entity.strip()) == 0:
            continue
        if len(text.lower()) == len(text) and len(entity.lower()) == len(entity):
            match_text, match_entity = text.lower(), entity.lower()
        else:
            match_text, match_entity = text, entity
        if not match_entity in match_text:
            continue
        match_entity_pattern = get_re_pattern(match_entity)
        for m in re.finditer(match_entity_pattern, match_text):
            start_idx, end_idx = m.start(), m.end() 
            if start_idx > 0 and text[start_idx-1].isalnum(): 
                continue
            if end_idx < len(text) and text[end_idx].isalnum():
                continue
            span_list.append((start_idx, end_idx, text[start_idx: end_idx], i))
    
    span_list = sorted(span_list, key=lambda x: (x[3], x[0]))
    filter_span_list = [] 
    for span in span_list:
        if not has_overlap_span(start_idx=span[0], end_idx=span[1], span_list=filter_span_list):
            filter_span_list.append(span[:3])

    return filter_span_list

def has_overlap_span(start_idx, end_idx, span_list):

    for span in span_list:
        span_start_idx, span_end_list = span[0], span[1]
        if end_idx <= span_start_idx or start_idx >= span_end_list:
            continue
        return True
    return False

def ner_using_spacy(text):

    entity_spans = []
    doc = nlp(text)

    noun_phrase_spans = []
    for ent in doc.ents:
        entity_spans.append((ent.start_char, ent.end_char, ent.text, ent.label_))
    
    noun_phrase_spans = sorted(noun_phrase_spans, key=lambda x: x[0])
    entity_spans = sorted(entity_spans, key=lambda x: x[0])
    
    i = 0 
    j = 0 
    spans = [] 
    while i < len(noun_phrase_spans) and j < len(entity_spans):
        noun_phrase_span_start = noun_phrase_spans[i][0]
        noun_phrase_span_end = noun_phrase_spans[i][1]
        entity_span_start = entity_spans[j][0]
        entity_span_end = entity_spans[j][1]
        if noun_phrase_span_end <= entity_span_start or noun_phrase_span_start >= entity_span_end:
            if noun_phrase_span_start < entity_span_start:
                span = noun_phrase_spans[i]
                i += 1 
            else:
                span = entity_spans[j]
                j += 1 
            spans.append((span[0], span[1], text[span[0]: span[1]], span[3]))
        else:
            if noun_phrase_span_end - noun_phrase_span_start > entity_span_end - entity_span_start:
                if noun_phrase_span_end == entity_span_end:
                    existing_noun_phrase_span = noun_phrase_spans[i]
                    noun_phrase_spans[i] = (existing_noun_phrase_span[0], existing_noun_phrase_span[1], existing_noun_phrase_span[2], entity_spans[j][3])
                j += 1 
            else:
                i += 1
    while i < len(noun_phrase_spans):
        span = noun_phrase_spans[i]
        spans.append((span[0], span[1], text[span[0]: span[1]], span[3]))
        i += 1 
    
    while j < len(entity_spans):
        span = entity_spans[j]
        spans.append((span[0], span[1], text[span[0]: span[1]], span[3]))
        j += 1 

    return spans 

def ner_pipeline_spacy(predefined_entities, text, question_entities=None):

    entity_linking_results = process_tagme_one_item(text, threshold=0.25)
    time.sleep(2)

    entity_linking_spans = [
        (span[0], span[1], mention, "PROPN", entity_name) for span, mention, entity_name in \
            zip(entity_linking_results["span"], entity_linking_results["mention"], entity_linking_results["entity_name"])
    ]
    text_spans = entity_linking_spans

    if predefined_entities is not None:
        predefined_entities_spans = get_entities_spans(predefined_entities, text=text)
        predefined_entities_spans = [(s, e, t, "PROPN", None) for s, e, t in predefined_entities_spans]
    else:
        predefined_entities_spans = []
    text_spans += [item for item in predefined_entities_spans if not has_overlap_span(item[0], item[1], text_spans)]

    if question_entities is not None:
        question_entity_spans = get_entities_spans(question_entities, text)
        question_entity_spans = [(s, e, t, "PROPN", None) for s, e, t in question_entity_spans]
    else:
        question_entity_spans = [] 
    text_spans += [item for item in question_entity_spans if not has_overlap_span(item[0], item[1], text_spans)]

    spacy_spans = ner_using_spacy(text)
    spacy_spans = [(s, e, t, et, None) for s, e, t, et in spacy_spans]
    text_spans += [item for item in spacy_spans if not has_overlap_span(item[0], item[1], text_spans)]

    text_spans = sorted(text_spans, key=lambda x: x[0])

    new_spans = []
    for span in text_spans:
        span_start_idx, span_end_idx, span_text, span_type, entity_name = span
        words = span_text.split()
        if words[0].lower() in set(["the", "a", "an"] + list(string.punctuation)):
            new_span_text = span_text[len(words[0]):].strip()
        else:
            new_span_text = span_text

        new_span_start_idx = span_end_idx - len(new_span_text)
        new_spans.append((new_span_start_idx, span_end_idx, new_span_text, span_type, entity_name))

    spans = new_spans

    return spans

def add_entity(mention, entity_name, entityid2mentions, normalizedmention2entityid):

    def lemmatize(words):
        words = words.lower()
        new_words = []
        for token in nlp(words):
            new_words.append(token.lemma_)
        return " ".join(new_words)

    entity_id = None
    entity_found = False

    for entity in [entity_name, mention]:
        if entity is None:
            continue
        if len(entity.strip()) == 0:
            continue
        compare_entity = lemmatize(entity)
        if compare_entity in normalizedmention2entityid:
            entity_id = normalizedmention2entityid[compare_entity]
            entity_found = True

        if entity_found:
            break
    
    if entity_found:
        return entity_id
    else:
        if len(mention.strip()) == 0:
            return None
        entity = entity_name if entity_name is not None else mention
        entity_id = len(entityid2mentions)
        entityid2mentions[entity_id].append(entity)
        normalizedmention2entityid[lemmatize(entity)] = entity_id
        return entity_id
        
def ner_spacy_tagme(item):

    """
    Input Format: 
    {
        "question": str, 
        "ctxs": {
            "id": str, 
            "title": str, 
            "text": str, 
        },
    }
    """

    entityid2mentions = defaultdict(list)
    normalizedmention2entityid = {}

    ctxs_list = [] 
    title_entities = [ctx["title"] for ctx in item["ctxs"] if "title" in ctx]
    supporting_facts = item["supporting_facts"] if "supporting_facts" in item else []
    predefined_entitities = supporting_facts + title_entities

    question_spans = ner_pipeline_spacy(predefined_entitities, item["question"])

    question_entities = []
    for span in question_spans:
        mention, entity_name = span[2], span[4]
        entity_id = add_entity(mention, entity_name, entityid2mentions, normalizedmention2entityid)
        if entity_id is not None and mention not in question_entities:
            question_entities.append(mention)

    for ctx_id, ctx in enumerate(item["ctxs"]):

        if "title" in ctx:
            title = ctx["title"].strip()
            title_entity_id = add_entity(title, None, entityid2mentions, normalizedmention2entityid)
            if title_entity_id is not None:
                new_title = "<e> {} </e>".format(title)
                title_entity = [(0, len(title), title, title_entity_id)]
            else:
                new_title = title
                title_entity = []

        # 识别text中的entity
        text = ctx["text"].strip()
        entity_span = ner_pipeline_spacy(predefined_entitities, text, question_entities=question_entities) 

        text_entity = []
        if len(entity_span) > 0:
            new_text = text[: entity_span[0][0]]
            for i, (start_idx, end_idx, mention, entity_type, entity_name) in enumerate(entity_span):
                entity_id = add_entity(mention, entity_name, entityid2mentions, normalizedmention2entityid)
                if entity_id is not None:
                    new_text = new_text + "<e> {} </e>".format(mention)
                    new_start_idx = start_idx+len(text_entity)*9+4
                    text_entity.append((new_start_idx, new_start_idx+len(mention), mention, entity_id))
                else:
                    new_text = new_text + mention
                if i < len(entity_span) - 1:
                    new_text = new_text + text[end_idx: entity_span[i+1][0]]
            new_text = new_text + text[entity_span[-1][1]:]
        else: 
            new_text = text

        one_ctx = {"id": ctx["id"]}
        if "title" in ctx:
            one_ctx["title"] = new_title
            one_ctx["title_entity"] = title_entity
        one_ctx["text"] = new_text
        one_ctx["text_entity"] = text_entity
        ctxs_list.append(one_ctx)

        # ctxs_list.append(
        #     {
                
        #         "title": new_title, 
        #         "text": new_text,
        #         "title_entity": title_entity,
        #         "text_entity": text_entity
        #     }
        # )
    
    new_ctxs_list = ctxs_list

    return {
        "ctxs": new_ctxs_list, 
        "question_entity": question_entities, 
        "entityid2name": entityid2mentions,
    }

def get_wikidata_entity_id_based_on_name(name):

    import time 
    import requests

    max_retries = 10 
    retry_delay = 60

    url = 'https://www.wikidata.org/w/api.php'
    params = {'action': 'wbsearchentities', 'language': 'en', 'format': 'json', 'search': name}

    for attempt in range(max_retries):
        response = requests.get(url, params=params)
        if response.status_code == 200:
            results = response.json().get("search")
            if results:
                id = results[0].get("id")
                wiki_name = results[0].get("label")
            else:
                id = "#UNK#"
                wiki_name = "#UNK#"
            break
        elif response.status_code == 403 or response.status_code == 429:
            print(f"Request Status Code: {response.status_code}.")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2 
            else:
                print(f"Max retries exceeded. Could not establish a connection!")
            id = "#UNK#"
            wiki_name = "#UNK#"
        else:
            id = "#UNK#"
            wiki_name = "#UNK#"
            break

    return id, wiki_name

def get_wikidata_entity_neighbors_based_on_entity_id(eid):

    import time 
    import requests

    max_retries = 10 
    retry_delay = 60

    eid = eid.strip()

    url = "https://www.wikidata.org/w/api.php"
    params = {"action": "wbgetclaims", "format": "json", "entity": eid}
    
    for attempt in range(max_retries):
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            neighbors = {}
            if "claims" in data:
                for property_id, claims in data["claims"].items():
                    for claim in claims:
                        main_snak = claim.get("mainsnak", {})
                        if main_snak.get("snaktype") == "value":
                            data_value = main_snak.get("datavalue", {}).get("value", {})
                            if isinstance(data_value, dict) and "id" in data_value:
                                neighbor_id = data_value["id"]
                                neighbors.setdefault(property_id, []).append(neighbor_id)
            break

        elif response.status_code == 403 or response.status_code == 429:
            print(f"Request Status Code: {response.status_code}.")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2 
            else:
                print(f"Max retries exceeded. Could not establish a connection!")
            neighbors = {}
        else:
            neighbors = {}
            break

    return neighbors

def get_triples_from_wikidata(item):

    entityid2wikidataid = {}
    for entity_id in item["entityid2name"]:
        for name in item["entityid2name"][entity_id]:
            wikidataid, wikidataname = get_wikidata_entity_id_based_on_name(name)
            if wikidataid == "#UNK#":
                continue
            break
        entityid2wikidataid[entity_id] = wikidataid

    wikidataid2entityid = {v: k for k, v in entityid2wikidataid.items() if v != "#UNK#"}
    triples = []
    for wikidataid in wikidataid2entityid.keys():
        neighbors = get_wikidata_entity_neighbors_based_on_entity_id(wikidataid)
        for relation, neighbor_wikidataids in neighbors.items():
            for weid in neighbor_wikidataids:
                if weid in wikidataid2entityid and wikidataid != weid:
                    triples.append((wikidataid2entityid[wikidataid], relation, wikidataid2entityid[weid]))

    item["entityid2wikidataid"] = entityid2wikidataid
    item["triples"] = triples

    return item 
