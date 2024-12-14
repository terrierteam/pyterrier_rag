import pickle 
from pandas import DataFrame
import os 
os.environ["JAVA_HOME"] = "/nfs/java/jdk-11.0.1"
import pyterrier as pt 

import ir_measures

from src.models import (
    REANO, 
    DocuNetRelationExtraction, 
    EntityRecognition,
    WikidataRelationExtraction
)
from src.util import load_json
from src.evaluation import f1_score, ems

# load data 
data = load_json("/nfs/common/data/2wikimultihopqa/reranker_data/test.json", type="json")
df_data = DataFrame.from_records(data=data)[:2]
df_data["qid"] = df_data["id"]
df_data["query"] = [question.replace("?", "").replace("-", "").replace("(", "").replace(")", "").replace(",", "") for question in df_data["question"]]
df_data["gold_answer"] = [ans_list[0] for ans_list in df_data["answers"]]
qrels = df_data[["qid", "gold_answer"]]
df_data = df_data[["qid", "query"]]

# load retriever model 
index = pt.IndexFactory.of("/nfs/kg_search/data/index/trec_dl_2019_judged")
dataset = pt.get_dataset("irds:msmarco-passage/trec-dl-hard") 
bm25 = pt.terrier.Retriever(index, wmodel="BM25") % 10 >> pt.text.get_text(dataset, "text")

# entity recognition 
entity_recognition = EntityRecognition()

# Wikidata Relation Extration 
wikidata_relation_extraction = WikidataRelationExtraction()

# DocuNet relation extraction 
docunet = DocuNetRelationExtraction(model_name_or_path="Jinyuan6/docunet")

reano_model = REANO(
    model_name_or_path="Jinyuan6/reano_2wiki", 
    docunet_model_name_or_path="Jinyuan6/docunet",
)

reano_pipe = (
    bm25
    >> entity_recognition
    >> wikidata_relation_extraction
    >> docunet
    >> pt.apply.triples(lambda row: list(set(row["wikidata_triples"]+row["docunet_pred_triples"])))
    >> reano_model
)
reano_output = reano_pipe.transform(df_data)

f1_measure = ir_measures.define_byquery(lambda qrels, res: f1_score(res.iloc[0]['answer'], qrels.iloc[0]['gold_answer']), support_cutoff=False, name="F1")
ems_measure = ir_measures.define_byquery(lambda qrels, res: ems(res.iloc[0]['answer'], qrels.iloc[0]['gold_answer']), support_cutoff=False, name="Exact Match")
metrics = pt.Evaluate(reano_output, qrels, [f1_measure, ems_measure])
print("metrics: ", metrics)


# print("loading data for relation extraction")
# data = pickle.load(open("/nfs/common/data/2wikimultihopqa/reano_data/test_with_triples.pkl", "rb"))
# df_data = DataFrame.from_records(data=data)[:10]
# df_data["gold_answer"] = df_data["answers"]
# df_data["qid"] = list(range(len(df_data)))

# docunet = DocuNetRelationExtraction(
#     model_path="/nfs/reano/checkpoints/relation_extraction/docunet.ckpt", 
#     relation_path="/nfs/reano/rebel_dataset", 
# )

# df_data = docunet.transform(df_data)

# merge wikidata & DocuNet relation
# df_data["triples"] = df_data.apply(lambda row: list(set(row["wikidata_triples"]+row["docunet_pred_triples"])), axis=1)

# print("loading data for answer generation & evaluation ...")
# data = pickle.load(open("/nfs/common/data/2wikimultihopqa/reano_data/test_with_relevant_triples_wounkrel.pkl", "rb"))
# df_data = DataFrame.from_records(data=data)[:3]
# df_data["gold_answer"] = df_data["answers"]
# df_data["qid"] = list(range(len(df_data)))
# df_data = pickle.load(open("test.pkl", "rb"))

# reano_model = REANO(
#     model_path="/nfs/reano/checkpoints/2wikimultihopqa", 
#     relation_path="/nfs/reano/rebel_dataset", 
# )

# metrics = ir_measures.calc_aggregate([EM, F1], qrels=df_data[["qid", "gold_answer"]], run=results)
# print(metrics)