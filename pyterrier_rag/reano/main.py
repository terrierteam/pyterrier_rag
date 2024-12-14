
import os
import glob
import pickle
import logging 
from pathlib import Path

import torch
import torch.distributed as dist
from src.options import Options

import transformers

from src.options import Options
from src.datasets import TripleFiDDataset, FiDCollator
from src.fid import TripleKGFiDT5
from src.trainer import ReaderTrainer
from src.util import gpu_setup, cleanup, get_dataloader, setup_logger

logger = logging.getLogger(__name__)


def main():

    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    opt = options.parse()

    gpu_setup(opt.local_rank, opt.seed) 

    model_name = 't5-' + opt.model_size
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)

    tokenizer.add_special_tokens({'sep_token': '[SEP]', 'cls_token': '[CLS]'})
    tokenizer.add_tokens(["<e>", "</e>"])

    # relation2id = pickle.load(open("/nfs/common/data/rebel_dataset/relation2id.pkl", "rb"))
    relation2id = pickle.load(open(opt.relation2id, "rb"))
    collator = FiDCollator(opt.text_maxlength, tokenizer=tokenizer, \
        relation2id=relation2id, answer_maxlength=opt.answer_maxlength, max_num_edge=opt.k)

    if not opt.test_only:

        train_dataset = TripleFiDDataset(opt.train_data, opt.n_context)
        val_dataset = TripleFiDDataset(opt.eval_data, opt.n_context)

        train_dataloader = get_dataloader(opt.local_rank, train_dataset, opt.per_gpu_batch_size, shuffle=True, collate_fn=collator)
        val_dataloader = get_dataloader(opt.local_rank, val_dataset, opt.per_gpu_batch_size, shuffle=False, collate_fn=collator)
        
    model = TripleKGFiDT5.from_pretrained(model_name, tokenizer=tokenizer, ent_dim=opt.ent_dim, k=opt.k, hop=opt.hop, alpha=opt.alpha, num_triples=opt.num_triples)
    model.resize_token_embeddings(len(tokenizer))
    relationid2name = pickle.load(open(opt.relationid2name, "rb"))
    relation_embedding = pickle.load(open(opt.init_relation_embedding, "rb"))
    model.relation_extraction_setup(
        relationid2name=relationid2name,
        relation_embedding=relation_embedding
    )

    dir_path = Path(opt.checkpoint_dir)/opt.name
    dir_path.mkdir(parents=True, exist_ok=True)
    options.print_options(opt)

    setup_logger(opt.local_rank, os.path.join(dir_path, "trainer.log"))
    
    trainer_params = {
        "tokenizer": tokenizer, 
        "model": model, 
        "local_rank": opt.local_rank, 
        "world_size": dist.get_world_size() if opt.local_rank >=0 else 1, 
        "default_root_dir": dir_path, 
        "learning_rate": opt.lr, 
        "weight_decay": opt.weight_decay, 
        "optim": "adamw", 
        "scheduler": opt.scheduler, 
        "warmup_steps": 0.2 * opt.total_steps, 
        "max_steps": opt.total_steps, 
        "accumulate_grad_batches": opt.accumulation_steps, 
        "gradient_clip_val": opt.clip, 
        "val_every_n_steps": opt.eval_freq, 
        "log_every_n_steps": 1, 
        "save_checkpoint_every_n_steps": opt.save_freq,
        "best_val_topk": 1, 
        "debug": not opt.use_wandb,
        "wandb_project": opt.wandb_project, 
        "wandb_name": opt.wandb_name,
        "pretrain": opt.pretrain, 
        "mask_passages": opt.mask_passages, 
        "num_passages_after_mask": opt.num_passages_after_mask
    }

    trainer = ReaderTrainer(**trainer_params)

    if not opt.test_only:
        trainer.train(train_dataloader, val_dataloader)

    if opt.local_rank <= 0:

        print("Loading test data from test_spacy.json ... ")
        test_file_name = os.path.basename(opt.eval_data).replace("dev", "test")
        test_data_path = os.path.join(os.path.dirname(opt.eval_data), test_file_name)

        test_dataset = TripleFiDDataset(test_data_path, opt.n_context)
        test_dataloader = get_dataloader(-1, test_dataset, opt.per_gpu_batch_size, shuffle=False, collate_fn=collator)

        for ckpt_path in glob.glob(os.path.join(dir_path, "best_val_*.ckpt")) + glob.glob(os.path.join(dir_path, "checkpoint_*.ckpt")):
            trainer.load_model_checkpoint(ckpt_path)
            metrics = trainer.evaluate(test_dataloader)
            logger.info(" ==== Test Results of {} ====".format(ckpt_path))
            logger.info("Exact Match: {:.2f}".format(100 * metrics["_main_eval_metric"]))
            logger.info("Avg Generate Time: {}".format(metrics.get("time", "NaN")))
            logger.info(" =======================================================")


    if opt.local_rank <=0 and opt.pretrain:
        ckpt_path = list(glob.glob(os.path.join(dir_path, "best_val_*.ckpt")))[0]
        new_ckpt_path = os.path.join(os.path.dirname(ckpt_path), "pretrain_best_val.ckpt")
        os.rename(ckpt_path, new_ckpt_path)

    if opt.local_rank >= 0:
        dist.barrier()

    if opt.local_rank >=0:
        cleanup()

    
if __name__ == "__main__":
    main()