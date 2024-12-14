
# python -m relation_extraction.docunet_inference \
#     --relation2id /nfs/reano/rebel_dataset/relation2id.pkl \
#     --docunet_checkpoint /nfs/reano/checkpoints/relation_extraction/docunet.ckpt \
#     --data_folder /nfs/common/data/2wikimultihopqa/reano_data

# python main.py \
#     --train_data /path/to/train_with_relevant_triples_wounkrel.pkl \
#     --eval_data /path/to/dev_with_relevant_triples_wounkrel.pkl \
#     --relation2id /path/to/rebel_data/relation2id.pkl \
#     --relationid2name /path/to/rebel_data/relationid2name.pkl \
#     --init_relation_embedding /path/to/rebel_data/relation_t5base_embeddings.pkl \
#     --model_size base \
#     --text_maxlength 250 \
#     --k 20 \
#     --hop 3 \
#     --n_context 10 \
#     --name default_experiment \
#     --checkpoint_dir checkpoint \
#     --lr 1e-4 \
#     --scheduler fixed \
#     --weight_decay 0.01 \
#     --per_gpu_batch_size 8 \
#     --accumulation_steps 8 \
#     --clip 1.0 \
#     --total_steps 15000 \
#     --eval_freq 500 \
#     --save_freq 15000


# export PYTHONPATH="/nfs/ir_measures:$PYTHONPATH"

python main_pyterrier.py

# /nfs/anaconda3/envs/llama/bin/python -m src.models