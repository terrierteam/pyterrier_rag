import os 
import time 
import wandb

import shutil
from tqdm import trange, tqdm
from queue import PriorityQueue
from collections import defaultdict
from contextlib import nullcontext

import json
import math
import inspect
import logging 
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist 
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from src.evaluation import ems 

from transformers import get_linear_schedule_with_warmup, get_constant_schedule

logger = logging.getLogger(__file__)

class BaseTrainer(nn.Module):

    def __init__(self, model, local_rank=-1, world_size=1, default_root_dir="./", learning_rate=3e-5, weight_decay=1e-3, \
                 optim="adamw", scheduler="linear", warmup_steps=0, max_epochs=-1, max_steps=-1, accumulate_grad_batches=1, \
                    gradient_clip_val=-1.0, val_every_n_steps=2000, log_every_n_steps=500, save_checkpoint_every_n_steps=-1, \
                        best_val_topk=1, fp16=False, bf16=False, use_amp=False, use_peft=False, use_qlora=False, debug=False, custom_eval_batch=False, **kwargs):
    
        super().__init__()

        if local_rank >= 0:
            self.device = torch.device(f"cuda:{local_rank}")
            self.use_ddp = True
        else:
            self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
            self.use_ddp = False
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)
        self.local_rank = local_rank

        self.world_size = world_size

        self.model = model.to(self.device)
        self.model = model
        self.default_root_dir = default_root_dir
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optim = optim
        self.scheduler_name = scheduler
        self.warmup_steps = warmup_steps
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.accumulate_grad_batches = accumulate_grad_batches
        self.gradient_clip_val = gradient_clip_val
        self.val_every_n_steps = val_every_n_steps
        self.log_every_n_steps = log_every_n_steps
        self.save_checkpoint_every_n_steps = save_checkpoint_every_n_steps
        self.best_val_topk = best_val_topk 

        self.fp16 = fp16
        self.bf16 = bf16
        self.use_amp = use_amp
        self.use_peft = use_peft
        self.use_qlora = use_qlora
        if not self.use_amp and (self.fp16 or self.bf16):
            data_type = "torch.float16" if fp16 else "torch.bfloat16"
            logger.warning(f"Rank: {self.local_rank}, Using {data_type} training! Note that if you want to train the model with half precision, the learning rate might need to be lower than that of torch.float32 version!")
        if self.use_amp:
            assert self.fp16 or self.bf16
            if self.fp16 and self.bf16:
                self.bf16 = False
                logger.warning(f"Rank: {self.local_rank}, Setting both fp16 and bf16 to True, will use fp16 by default")
            logger.warning(f"""Rank: {self.local_rank}, Using mix precision training! Note that NaN gradients can sometimes happen, but NaN loss is not expected, you should can check your code when loss is NaN. If there is NaN in loss, you can try the following: (1) use torch.bfloat16 instead of torch.float16. (2) convert data type to torch.float32 when calculating loss (such as torch.log(x.float()+1e-9) ect.). (3) use a smaller learning rate. (4) set the eps value in AdamW from 1e-08 to 1e-03 """) 
        if self.use_qlora:
            logger.warning(f"Rank: {self.local_rank}, Using QLoRA.")
        self.custom_eval_batch = custom_eval_batch
        if self.custom_eval_batch:
            logger.warning(f"Rank: {self.local_rank}, Setting custom_eval_batch to True, you need to move tensors to GPU yourself in evaluation_step() function, otherwise it will cause error.")

        self.debug = debug
        self.kwargs = kwargs 
        self.configs = self.get_parameters_dict(ignore=["device", "model"], save_config=True)

        self.additional_init_setup(**kwargs)

    def additional_init_setup(self, **kwargs):
        pass

    def get_parameters_dict(self, ignore=None, save_config=True):

        params = self.__dict__
        ignore = [] if ignore is None else ignore
        configs = {}
        for k, v in params.items():
            if k in ignore or k.startswith("_"):
                continue
            if k == "kwargs":
                for kk, vv in params[k].items():
                    if kk in ignore:
                        continue
                    configs[kk] = vv 
                continue
            configs[k] = v 
        
        if save_config:
            with open(f"{self.default_root_dir}/configs.json", "w", encoding="utf-8") as fout:
                default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
                fout.write(json.dumps(configs, default=default, indent=4))

        return configs

    def setup_model(self, model):
        if self.use_ddp:
            return DDP(model.to(self.device), device_ids=[self.local_rank], output_device=self.local_rank) 
        else:
            return model.to(self.device)
    
    def setup_batch(self, batch):

        if isinstance(batch, (tuple, list)):
            new_batch = [] 
            for item in batch:
                if isinstance(item, dict):
                    new_batch.append(self.setup_dict_batch_data(item))
                elif torch.is_tensor(item):
                    new_batch.append(item.to(self.device))
                else:
                    new_batch.append(item)
        elif isinstance(batch, dict):
            new_batch = self.setup_dict_batch_data(batch)
        else:
            raise TypeError(f"Currently do not support using <{type(batch)}> as the type of a batch")

        return new_batch

    def setup_dict_batch_data(self, data):
        return {k: item.to(self.device) if torch.is_tensor(item) else item for k, item in data.items()}

    def average_main(self, value, main_process, use_ddp):
                    
        if not use_ddp:
            return value
        if self.world_size > 1:
            dist.reduce(value, dst=0, op=dist.ReduceOp.SUM)
            if main_process:
                value = value / self.world_size

        return value
    
    def sum_main(self, value, main_process, use_ddp):
        if not use_ddp:
            return value
        if self.world_size > 1:
            dist.reduce(value, dst=0, op=dist.ReduceOp.SUM)
        return value 

    def weighted_average(self, avg_value, count, main_process, use_ddp):

        if not use_ddp:
            return avg_value
        
        avg_value = torch.tensor([avg_value], device=self.device)
        count = torch.tensor([count], device=self.device)

        if main_process:
            gathered_avg_value = [torch.zeros_like(avg_value) for _ in range(self.world_size)] 
            gathered_count = [torch.zeros_like(count) for _ in range(self.world_size)] 
            dist.gather(avg_value, dst=0, gather_list=gathered_avg_value)
            dist.gather(count, dst=0, gather_list=gathered_count)
            value_across_gpu = gathered_avg_value[0] * gathered_count[0]
            count_across_gpu = gathered_count[0]
            for avg, c in zip(gathered_avg_value[1:], gathered_count[1:]):
                value_across_gpu += avg * c 
                count_across_gpu += c 
            avg_value = value_across_gpu / count_across_gpu
        else:
            dist.gather(avg_value, dst=0)
            dist.gather(count, dst=0)

        return avg_value.item()

    def get_time_format(self, interval_seconds):
        days, remainder = divmod(interval_seconds, 3600*24)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        days = int(days)
        hours = int(hours)
        minutes = int(minutes)
        if days > 0:
            t = "{}d:{}h:{}m:{:.1f}s".format(days, hours, minutes, seconds)
        elif hours > 0:
            t = "{}h:{}m:{:.1f}s".format(hours, minutes, seconds)
        elif minutes > 0:
            t = "{}m:{:.1f}s".format(minutes, seconds)
        else:
            t = "{:.1f}s".format(seconds)

        return t 

    def get_max_training_steps(self, train_dataloader):

        assert self.max_epochs > 0 or self.max_steps > 0 
        if self.max_epochs > 0 and self.max_steps >0:
            logger.info(f"Rank: {self.local_rank}, Both max_epochs and max_steps are set, use only max_steps.")

        if self.max_steps < 0:
            num_batches_in_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = num_batches_in_dataloader // self.accumulate_grad_batches
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            max_training_steps = math.ceil(self.max_epochs * num_update_steps_per_epoch)
            logger.info(f"Rank: {self.local_rank}, Use max_epochs, the maximum number of training steps will be: {max_training_steps}")
        else: 
            max_training_steps = self.max_steps
        
        return max_training_steps
    
    def get_peft_checkpoint_path(self, ckpt_path):

        ckpt_path_parent = os.path.dirname(ckpt_path)
        save_folder = "peft_adapter_" + os.path.basename(ckpt_path)
        peft_ckpt_path = os.path.join(ckpt_path_parent, save_folder)
        return peft_ckpt_path
    
    def get_pretrained_checkpoint_path(self, ckpt_path):

        ckpt_path_parent = os.path.dirname(ckpt_path)
        save_folder = "pretrained_model_" + os.path.basename(ckpt_path)
        pretrained_ckpt_path = os.path.join(ckpt_path_parent, save_folder)
        return pretrained_ckpt_path

    def save_model_checkpoint(self, ckpt_path):

        best_val_topk_score = []
        best_val_topk_step = []
        best_val_topk_ckpt_name = []
        for i in range(self.best_val_topk_checkpoint.qsize()):
            val_score, val_step, ckpt_name = self.best_val_topk_checkpoint.queue[-1-i]
            best_val_topk_score.append(val_score)
            best_val_topk_step.append(val_step)
            best_val_topk_ckpt_name.append(ckpt_name)
        
        unwrap_model = self.model.module if hasattr(self.model, "module") else self.model

        if self.use_peft or self.use_qlora:
            saved_model_params = None
            unwrap_model.save_pretrained(self.get_peft_checkpoint_path(ckpt_path))
        else:
            saved_model_params = unwrap_model.state_dict()

        checkpoint = {
            "configs": self.configs,
            "model": saved_model_params,
            "optimizer": [opt.state_dict() for opt in self.optimizer] if isinstance(self.optimizer, (list,tuple)) else self.optimizer.state_dict(),
            "scheduler": [sch.state_dict() for sch in self.scheduler] if isinstance(self.scheduler, (list, tuple)) else self.scheduler.state_dict(), 
            "scaler": self.amp_scaler.state_dict() if self.use_amp else None,
            "current_epoch": self.current_epoch,
            "global_step": self.global_step, 
            "best_val_topk_score": best_val_topk_score,
            "best_val_topk_step": best_val_topk_step,
            "best_val_topk_ckpt_name": best_val_topk_ckpt_name
        }
        logger.info(f"Rank: {self.local_rank}, Saving checkpoint to {ckpt_path}")
        torch.save(checkpoint, ckpt_path)

        return ckpt_path
    
    def load_model_checkpoint(self, ckpt_path):

        logger.info(f"Rank: {self.local_rank}, Loading checkpoint from {ckpt_path}")
        unwrap_model = self.model.module if hasattr(self.model, "module") else self.model
        if self.use_peft or self.use_qlora:
            from my_utils import load_peft
            unwrap_model = load_peft(unwrap_model, self.get_peft_checkpoint_path(ckpt_path)) 
            unwrap_model = unwrap_model.float()
        else:
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            unwrap_model.load_state_dict(checkpoint["model"], strict=False)
            unwrap_model = unwrap_model.float() 
            newly_parameters = []
            for k in unwrap_model.state_dict().keys():
                if k not in checkpoint["model"]:
                    newly_parameters.append(k)
            if newly_parameters:
                logger.info("####################################################################")
                logger.info(f"Some parameters not initialized from checkpoint: {newly_parameters}")
                logger.info("####################################################################")
                    
        return unwrap_model
    
    def save_val_topk_checkpoint(self, ckpt_name, val_score):
        self.best_val_topk_checkpoint.put((val_score, self.global_step, ckpt_name))
        ckpt_save_path = os.path.join(self.default_root_dir, ckpt_name)
        ckpt_save_path = self.save_model_checkpoint(ckpt_save_path)
        return ckpt_save_path

    def best_val_topk_handler(self, ckpt_name, val_score):

        current_num_best_val_checkpoint = self.best_val_topk_checkpoint.qsize()
        if current_num_best_val_checkpoint < self.best_val_topk:
            return self.save_val_topk_checkpoint(ckpt_name, val_score)
        
        compare_score, _, _ = self.best_val_topk_checkpoint.queue[0]
        if val_score < compare_score:
            return None 
        
        _, _, pop_ckpt_name = self.best_val_topk_checkpoint.get()
        pop_ckpt_save_path = os.path.join(self.default_root_dir, pop_ckpt_name)
        if os.path.exists(pop_ckpt_save_path):
            if os.path.isdir(pop_ckpt_save_path):
                shutil.rmtree(pop_ckpt_save_path)
            else:
                os.remove(pop_ckpt_save_path)
            if self.use_peft or self.use_qlora:
                pop_peft_checkpoint_path = self.get_peft_checkpoint_path(pop_ckpt_save_path)
                if os.path.exists(pop_peft_checkpoint_path):
                    shutil.rmtree(pop_peft_checkpoint_path)

        return self.save_val_topk_checkpoint(ckpt_name, val_score)
                
    def resume_training(self, ckpt_path):

        logger.info(f"Rank: {self.local_rank}, Resume Training from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        unwrap_model = self.model.module if hasattr(self.model, "module") else self.model
        if checkpoint["model"] is not None:
            unwrap_model.load_state_dict(checkpoint["model"])

        if isinstance(self.optimizer, (list, tuple)):
            for i, (opt, sch) in enumerate(zip(self.optimizer, self.scheduler)):
                opt.load_state_dict(checkpoint["optimizer"][i])
                sch.load_state_dict(checkpoint["scheduler"][i])
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        
        if self.use_amp and checkpoint["scaler"] is not None:
            self.amp_scaler.load_state_dict(checkpoint["scaler"])

        if hasattr(self.scheduler, "last_epoch"):
            self.scheduler.last_epoch = checkpoint["global_step"]
        self.prev_step = checkpoint["global_step"]
        for val_score, val_step, ckpt_name in zip(checkpoint["best_val_topk_score"], \
            checkpoint["best_val_topk_step"], checkpoint["best_val_topk_ckpt_name"]):
            self.best_val_topk_checkpoint.put((val_score, val_step, ckpt_name))

        return checkpoint

    def log(self, name, value):
        caller_function = inspect.stack()[1].function 
        if caller_function == "training_step":
            self.log_tensor[name] = value 
        elif caller_function == "evaluate_step":
            assert isinstance(value, (tuple, list, np.ndarray)) 
            sum_batch_score, batch_size = self.calculate_batch_eval_score(value)
            self.log_evaluation_sum_score[name] = self.log_evaluation_sum_score.get(name, 0.0) + sum_batch_score
            self.log_evaluation_total[name] = self.log_evaluation_total.get(name, 0) + batch_size
        else:
            assert False 
    
    def calculate_batch_eval_score(self, batch_scores):
        if isinstance(batch_scores, tuple):
            assert len(batch_scores) == 2 
            assert isinstance(batch_scores[1], int) 
        if isinstance(batch_scores, tuple):
            return batch_scores[0] * batch_scores[1], batch_scores[1]
        return np.sum(batch_scores), len(batch_scores)

    def train(self, train_dataloader, val_dataloader=None, test_dataloader=None, ckpt_path=None):

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        if self.use_ddp:
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)

        self.optimizer, self.scheduler = self.configure_optimizer(self.model) 
        if isinstance(self.optimizer, (list, tuple)):
            assert len(self.optimizer) == len(self.scheduler) 
        
        if self.use_amp:
            self.amp_scaler = torch.cuda.amp.GradScaler(enabled=True)

        main_process = self.local_rank == 0 if self.use_ddp else True
        logger.info(f"Rank: {self.local_rank}, main_process: {main_process}")
        if not self.debug and main_process:
            wandb_log = True
        else:
            wandb_log = False
        if wandb_log:
            wandb.init(project=self.kwargs.get("wandb_project", "Default"), config=self.configs, \
                       name=self.kwargs.get("wandb_name", "default_experiment_name"))
        
        self.prev_step = 0 
        self.best_val_topk_checkpoint = PriorityQueue() 
        self.current_epoch, self.global_step = 0, 0 
        self.max_training_steps = self.get_max_training_steps(train_dataloader)

        if ckpt_path is not None:
            checkpoint = self.resume_training(ckpt_path)
            if main_process:
                logger.info(f"=========== Resume Training, Checkpoint info: ================")
                logger.info(f"Previous Steps: {self.prev_step}")
                logger.info("Best TopK Validation Scores: {}".format(checkpoint["best_val_topk_score"]))
                logger.info("Best TopK Valiation Steps: {}".format(checkpoint["best_val_topk_step"]))
                logger.info("Best TopK Checkpoint Name: {}".format(checkpoint["best_val_topk_ckpt_name"]))
                logger.info("==============================================================")

        train_start_time = time.time()
        while self.global_step < self.max_training_steps:

            self.current_epoch += 1 

            self.training_epoch_start(train_dataloader)

            logger.info(f"Rank: {self.local_rank}, Epoch: {self.current_epoch} Training ... ")
            if self.use_ddp:
                train_dataloader.sampler.set_epoch(self.current_epoch) 
                dist.barrier() 
            
            num_update_step_per_epoch = len(train_dataloader) // self.accumulate_grad_batches
            num_batch_per_epoch = num_update_step_per_epoch * self.accumulate_grad_batches
            if self.current_epoch == 1:
                logger.info(f"Rank: {self.local_rank}, number of batches in the dataloader: {len(train_dataloader)}, number of batches used in an epoch: {num_batch_per_epoch}")
            
            for batch_idx, batch in enumerate(train_dataloader):

                if batch_idx >= num_batch_per_epoch:
                    break

                if self.global_step < self.prev_step:
                    if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                        self.global_step += 1 
                        if val_dataloader is not None and self.global_step % self.val_every_n_steps == 0:
                            for _ in val_dataloader:
                                break
                    continue

                self.model.train()
                if batch_idx % self.accumulate_grad_batches == 0: 

                    optimizer_list = self.optimizer if isinstance(self.optimizer, (tuple, list)) else [self.optimizer]
                    for opt in optimizer_list:
                        opt.zero_grad()

                    step_start_time = time.time()
                    loss_per_update_step = 0.0 
                    self.log_variable_per_update_step = defaultdict(float)
                    self.log_tensor = {}

                    self.training_step_start()
                
                batch = self.setup_batch(batch)
                unwrap_model = self.model.module if self.use_ddp else self.model

                context = self.model.no_sync if self.use_ddp else nullcontext
                with context():
                    if self.use_amp:
                        autocast_dtype = torch.float16 if self.fp16 else torch.bfloat16
                        with torch.autocast(enabled=True, device_type=self.device.type, dtype=autocast_dtype):
                            batch_outputs = self.training_step(unwrap_model, batch)
                    else:
                        batch_outputs = self.training_step(unwrap_model, batch)
                    if not isinstance(batch_outputs, (tuple, list)):
                        batch_outputs = (batch_outputs, )
                    batch_loss = batch_outputs[0] / (self.accumulate_grad_batches * self.world_size)
                    if self.use_amp:
                        assert batch_loss.dtype == torch.float32 
                        self.amp_scaler.scale(batch_loss).backward()
                    else:
                        batch_loss.backward()
                if self.use_ddp and (batch_idx + 1) % self.accumulate_grad_batches == 0:
                    for param in self.model.parameters():
                        if param.grad is not None: 
                            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    dist.barrier()
                batch_loss = self.sum_main(batch_loss, main_process, self.use_ddp)
                loss_per_update_step += batch_loss.item()
                
                for log_n, log_t in self.log_tensor.items():
                    log_t = log_t / (self.accumulate_grad_batches * self.world_size)
                    log_t = self.sum_main(log_t, main_process, self.use_ddp)
                    self.log_variable_per_update_step[log_n] += log_t.item()
                
                if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                    if self.use_amp:
                        for opt in (self.optimizer if isinstance(self.optimizer, (list, tuple)) else [self.optimizer]):
                            self.amp_scaler.unscale_(opt)
                    # gradient clipping 
                    if self.gradient_clip_val > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                    
                    grad_status = self.compute_grad_status(grad_all_reduced=True)
                    if grad_status["has_inf_nan"]:
                        logger.info("Rank: {}, Detected Inf or NaN gradient at: {}. Skip gradient update at global step {}.".format(\
                            self.local_rank, grad_status["params_first_with_inf_nan"], self.global_step+1))
                        if self.use_amp:
                            self.amp_scaler.update()
                        for opt in (self.optimizer if isinstance(self.optimizer, (tuple, list)) else [self.optimizer]):
                            opt.zero_grad()
                    else:
                        self.optimizer_step(self.optimizer, self.scheduler)

                    self.global_step += 1 
                    if wandb_log:
                        wandb.log({"loss":loss_per_update_step, "grad_min": grad_status["grad_min"], "grad_max": grad_status["grad_max"], \
                                   "grad_mean": grad_status["grad_mean"], "time (s)": time.time()-step_start_time, \
                                   **self.log_variable_per_update_step}, step=self.global_step)
                    if main_process and self.global_step % self.log_every_n_steps == 0:
                        current_time = time.time()
                        one_step_time = current_time - step_start_time
                        trained_time = self.get_time_format(current_time-train_start_time)
                        estimated_time = self.get_time_format(one_step_time*(self.max_training_steps-self.global_step))
                        training_info = "Rank: {}, Epoch: {}, Step: {}/{}, loss: {:.5f}, grad_min: {:.5f}, grad_max: {:.5f}, grad_mean(%): {:.5f}, ".format(\
                            self.local_rank, self.current_epoch, self.global_step, self.max_training_steps, loss_per_update_step,\
                                grad_status["grad_min"], grad_status["grad_max"], 100*grad_status["grad_mean"])
                        for log_n, log_v in self.log_variable_per_update_step.items():
                            training_info += "{}: {:.5f}, ".format(log_n, log_v)
                        training_info += "time: {}/{}".format(trained_time, estimated_time)
                        logger.info(training_info)
                    
                    self.training_step_end()

                    # evaluate
                    if val_dataloader is not None and self.global_step % self.val_every_n_steps == 0:
                        val_metric_dict = self.evaluate(val_dataloader, main_process) 
                        if main_process:
                            val_metric = val_metric_dict.pop("_main_eval_metric")

                            val_info = "Rank: {}, Epoch: {}, Step: {}, Evaluation Score: {:.5f}".format(self.local_rank, self.current_epoch, self.global_step, val_metric)
                            for val_k, val_v in val_metric_dict.items():
                                val_info += ", {}: {:.5f}".format(val_k, val_v)
                            logger.info(val_info)
                            
                            ckpt_name = "best_val_epoch{}_step{}.ckpt".format(self.current_epoch, self.global_step)
                            self.best_val_topk_handler(ckpt_name, val_metric) 

                            # logging 
                            if wandb_log:
                                wandb.log(
                                    {
                                        "val_metric": val_metric, 
                                        "best_val_metric": self.best_val_topk_checkpoint.queue[-1][0],
                                        "best_val_step": self.best_val_topk_checkpoint.queue[-1][1],
                                        **val_metric_dict,
                                    }
                                )
                    
                    if self.use_ddp:
                        dist.barrier()

                    if main_process and self.save_checkpoint_every_n_steps > 0 and \
                        self.global_step % self.save_checkpoint_every_n_steps == 0:
                        ckpt_path = "{}/checkpoint_epoch{}_step{}.ckpt".format(\
                            self.default_root_dir, self.current_epoch, self.global_step)
                        self.save_model_checkpoint(ckpt_path)
                    
                    if self.use_ddp:
                        dist.barrier()

                    # stop training 
                    if self.global_step >= self.max_training_steps:
                        logger.info(f"Rank: {self.local_rank}, Reach maximum steps {self.max_training_steps}, Stopping ... ")
                        break

            self.training_epoch_end(train_dataloader)

        total_train_time = self.get_time_format(time.time()-train_start_time)
        logger.info(f"Rank: {self.local_rank}, Total Training Time: {total_train_time}")
        
    def evaluate(self, dataloader, main_process=True):
    
        use_ddp_val = isinstance(dataloader.sampler, DistributedSampler)
        if use_ddp_val:
            if main_process:
                logger.info("Use DDP for evaluation. Note that the results may not be accurate. It is recommended not to use DDP for test set evaluation.")
        else:
            if main_process:
                logger.info("Not use DDP for evaluation at main_process.")
            else:
                return None
            
        self.model.eval()

        self.evaluate_epoch_start(dataloader)

        total = 0 
        all_sum_score = 0.0
        self.log_evaluation_sum_score = {}
        self.log_evaluation_total = {}

        unwrap_model = self.model.module if hasattr(self.model, "module") else self.model
        if main_process:
            description = "(Using DDP, results are mean aggregated)" if use_ddp_val else ""
            progress_bar = tqdm(total=len(dataloader), desc="Evaluation, Rank: {} {}".format(self.local_rank, description))
        for batch_idx, batch in enumerate(dataloader):
            if not self.custom_eval_batch:
                batch = self.setup_batch(batch)
            self.evaluate_step_start()
            with torch.no_grad():
                batch_scores = self.evaluate_step(unwrap_model, batch, dataloader)
            assert isinstance(batch_scores, (tuple, list, np.ndarray)) # evaluate_step must return tuple (avg_val_metric, num_batch_samples) or list, np.ndarray [val_metric of each sample in the batch]
            sum_batch_score, batch_size = self.calculate_batch_eval_score(batch_scores)
            all_sum_score += sum_batch_score
            total += batch_size
            self.evaluate_step_end()

            if main_process:
                progress_bar.update(1)

        if main_process:
            progress_bar.close()

        score = self.weighted_average(all_sum_score / total, total, main_process, use_ddp_val)
        log_evaluation_score = {}
        for k in self.log_evaluation_sum_score.keys():
            log_evaluation_score[k] = self.weighted_average(
                self.log_evaluation_sum_score[k] / self.log_evaluation_total[k], 
                self.log_evaluation_total[k], main_process, use_ddp_val
            )
        self.evaluate_epoch_end(dataloader)

        return {"_main_eval_metric": score, **log_evaluation_score}

    def configure_optimizer(self, model):
        if self.use_qlora:
            return self.configure_optimizer_paged_adamw(model)
        else:
            return self.configure_optimizer_adamw_fp32(model)

    def configure_optimizer_adamw_fp32(self, model):

        logger.info(f"Rank: {self.local_rank}, Using default Transformers AdamW optimizer 32bit with {self.scheduler_name} scheduler!")
        from transformers import AdamW
        optimizer = AdamW(self.get_optimizer_grouped_parameters(model), lr=self.learning_rate)
        scheduler = self.get_scheduler(optimizer=optimizer)
        return optimizer, scheduler

    def configure_optimizer_paged_adamw(self, model):
        logger.info(f"Rank: {self.local_rank}, Using Bitsandbytes Paged AdamW optimizer 32bit with {self.scheduler_name} scheduler!!")
        from bitsandbytes.optim import AdamW 
        adam_kwargs = {"betas": (0.9, 0.999), "eps": 1e-08, "lr": self.learning_rate, 'is_paged': True, 'optim_bits': 32}
        optimizer = AdamW(self.get_optimizer_grouped_parameters(model), **adam_kwargs)
        scheduler = self.get_scheduler(optimizer=optimizer)
        return optimizer, scheduler

    def get_scheduler(self, optimizer):

        if "linear" in self.scheduler_name:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.max_steps,
            )
        elif "constant" in self.scheduler_name or "fixed" in self.scheduler_name:
            scheduler = get_constant_schedule(optimizer)
        else:
            raise NotImplemented(f"{self.scheduler_name} is currently not implemented!")

        return scheduler

    def get_optimizer_grouped_parameters(self, model, no_decay_params=[]):
        assert isinstance(no_decay_params, (tuple, list)) # no_decay_params must be list or tuple
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.weight"] + list(no_decay_params)
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": self.weight_decay,},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and p.requires_grad],"weight_decay": 0.0},
        ]

        return optimizer_grouped_parameters
    
    def optimizer_step(self, optimizer, scheduler):

        if isinstance(optimizer, (list, tuple)):
            optimizer_list = optimizer
            scheduler_list = scheduler
        else:
            optimizer_list = [optimizer]
            scheduler_list = [scheduler]
        
        if self.use_amp:
            for opt in optimizer_list:
                self.amp_scaler.step(opt)
            self.amp_scaler.update()
            for opt, sch in zip(optimizer_list, scheduler_list):
                sch.step()
                opt.zero_grad()
        else:
            for opt, sch in zip(optimizer_list, scheduler_list):
                opt.step()
                sch.step()
                opt.zero_grad()
    
    def compute_grad_status(self, grad_all_reduced=False):

        with torch.no_grad():
            stats = [] 
            params_with_inf_nan = [] 
            unwrap_model = self.model.module if self.use_ddp else self.model
            for n, p in unwrap_model.named_parameters():
                if p.grad is not None:
                    grad_abs = torch.abs(p.grad)
                    grad_min = torch.min(grad_abs)
                    grad_max = torch.max(grad_abs)
                    grad_mean = torch.mean(grad_abs)
                    stats += [grad_min.item(), grad_max.item(), grad_mean.item()]
                    if (torch.isnan(grad_mean) or torch.isinf(grad_mean)) and len(params_with_inf_nan)==0:
                        params_with_inf_nan.append(n)
                else:
                    stats += [0.0, 0.0, 0.0]
            stats = torch.Tensor(stats).to(self.device)
            if self.use_ddp and not grad_all_reduced:
                dist.all_reduce(stats)
            stats = stats.view(-1, 3)
            grad_has_inf_nan = (torch.any(torch.isinf(stats)) or torch.any(torch.isnan(stats))).item()

            result = {}
            result["has_inf_nan"] = grad_has_inf_nan
            result["params_first_with_inf_nan"] = params_with_inf_nan
            result["grad_min"] = stats.min(0)[0][0].item()
            result["grad_max"] = stats.max(0)[0][1].item()
            result["grad_mean"] = stats.mean(0)[2].item()

            return result

    def training_step(self, model, batch):
        pass 

    def evaluate_step(self, model, batch, dataloader):
        pass

    def training_epoch_start(self, dataloader, **kwargs):
        pass 

    def training_epoch_end(self, dataloader, **kwargs):
        pass

    def training_step_start(self, **kwargs):
        pass

    def training_step_end(self, **kwargs):
        pass 

    def evaluate_epoch_start(self, dataloader, **kwargs):
        pass

    def evaluate_epoch_end(self, dataloader, **kwargs):
        pass

    def evaluate_step_start(self, **kwargs):
        pass

    def evaluate_step_end(self, **kwargs):
        pass


class ReaderTrainer(BaseTrainer):


    def __init__(self, **kwargs):
        
        self.tokenizer = kwargs.pop("tokenizer")
        super().__init__(**kwargs)
        self.mask_passages = kwargs.get("mask_passages", False)
        self.num_passages_after_mask = kwargs.get("num_passages_after_mask", 10)
        if self.mask_passages:
            logger.info(f"---->> mask passages is True, will mask passages during evaluation and only use {self.num_passages_after_mask} passages!")

    def training_step(self, model, batch):
        
        (idx, labels, _, context_ids, context_mask, question_text, question_indices, question_mask, \
            ent_indices, ent_mask, ent_is_ans, entity_text, entity_adj, entity_adj_mask, \
                entity_adj_relation, entity_adj_relevant_relation_label, passage_entity_ids, passage_entity_mask) = batch
        
        calculate_ans_loss = True
        calculate_kg_loss = True

        train_loss = model(
            input_ids=context_ids, 
            attention_mask = context_mask, 
            question_indices=question_indices,
            question_mask=question_mask,
            ent_indices=ent_indices,
            ent_mask=ent_mask,
            entity_text=entity_text, 
            entity_adj=entity_adj, 
            entity_adj_mask=entity_adj_mask, 
            entity_adj_relation=entity_adj_relation,
            labels = labels, 
            entity_adj_relevant_relation_label = entity_adj_relevant_relation_label, 
            ent_is_ans_label=ent_is_ans,
            calculate_ans_loss=calculate_ans_loss,
            calculate_kg_loss=calculate_kg_loss,
            question_text=question_text, 
        )[0]

        return train_loss
    
    def evaluate_step(self, model, batch, dataloader):

        dataset = dataloader.dataset 

        (idx, labels, _, context_ids, context_mask, question_text, question_indices, question_mask, \
            ent_indices, ent_mask, ent_is_ans, entity_text, entity_adj, entity_adj_mask, \
                entity_adj_relation, entity_adj_relevant_relation_label, passage_entity_ids, passage_entity_mask) = batch
        
        mask_passages = self.mask_passages
        num_passages_after_mask = self.num_passages_after_mask 
        
        outputs = model.generate(
            input_ids=context_ids,
            attention_mask=context_mask,
            question_indices=question_indices,
            question_mask=question_mask,
            ent_indices=ent_indices,
            ent_mask=ent_mask,
            entity_text=entity_text, 
            entity_adj=entity_adj, 
            entity_adj_mask=entity_adj_mask, 
            entity_adj_relation=entity_adj_relation, 
            max_length=50,
            question_text=question_text, 
            mask_passages = mask_passages, 
            num_passages_after_mask = num_passages_after_mask, 
            passage_entity_ids = passage_entity_ids, 
            passage_entity_mask = passage_entity_mask
        )

        score_list = []
        for i, o in enumerate(outputs):
            ans = self.tokenizer.decode(o, skip_special_tokens=True)
            gold = dataset.get_example(idx[i])["answers"]
            score = ems(ans, gold)
            score_list.append(score)

        return score_list
