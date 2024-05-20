from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
import transformers
import torch
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
import json
import pandas as pd
import argparse
import os
from omegaconf import OmegaConf


class LapisTrainer(object):
    def __init__(self, conf, logging):
        self.logging            = logging

        path_dataset            = os.path.join(conf.path.dataset, conf.dataprep.finetuning_dataset)
        self.path_dataset_train = os.path.join(path_dataset, f"{conf.dataprep.instruction_method}_train_{conf.dataprep.subsample}.hf")
        self.path_dataset_dev   = os.path.join(path_dataset, f"{conf.dataprep.instruction_method}_dev_{conf.dataprep.subsample}.hf")

        os.environ["WANDB_API_KEY"]   = conf.wandb.api_key
        os.environ["WANDB_JOB_TYPE"]  = 'training'
        os.environ["WANDB_PROJECT"]   = conf.wandb.project_name
        os.environ["WANDB_RUN_GROUP"] = conf.wandb.group_name
        os.environ["WANDB_NAME"]      = conf.wandb.session_name
        os.environ["WANDB_LOG_MODEL"] = 'checkpoint'
        wandb_name              = f'{conf.wandb.project_name}_{conf.wandb.group_name}_{conf.wandb.session_name}/'
        self.checkpoint_path    = os.path.join(conf.path.checkpoint, wandb_name)
        if not os.path.isdir(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)
            OmegaConf.save(config=conf, f=os.path.join(self.checkpoint_path, 'config.yaml'))

        self.llm_backbone       = conf.finetune.llm_backbone
        self.use_lora           = conf.finetune.lora.enabled
        if 'Polyglot' in conf.finetune.llm_backbone: 
            lora_modules   = ['query_key_value']
        else: 
            lora_modules   = ["q_proj", "k_proj", "v_proj", "o_proj"]
        self.lora_config   = LoraConfig(
                            r=conf.finetune.lora.r,
                            lora_alpha=conf.finetune.lora.alpha,
                            target_modules=lora_modules, # depends on module
                            lora_dropout=conf.finetune.lora.dropout,
                            bias=conf.finetune.lora.bias,
                            task_type=conf.finetune.lora.task_type)
        self.train_config  = transformers.TrainingArguments(
                            per_device_train_batch_size=conf.finetune.per_device_train_batch_size,
                            gradient_accumulation_steps=conf.finetune.gradient_accumulation_steps,
                            num_train_epochs=conf.finetune.num_train_epochs,
                            learning_rate=conf.finetune.learning_rate,
                            fp16=conf.finetune.fp16,
                            logging_steps=conf.finetune.logging_steps,
                            output_dir=self.checkpoint_path,
                            optim=conf.finetune.optim,
                            optim_target_modules=["attn", "mlp"] if conf.finetune.optim=='galore_adamw' else None,
                            report_to=conf.finetune.report_to,
                            save_strategy="epoch",
                            evaluation_strategy="epoch")

        self.logging.info(self.lora_config)
        self.logging.info(self.train_config)

    @torch.no_grad()
    def print_trainable_parameters(self, model):
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        self.logging.info(f"Trainable: {trainable_params} || Total: {all_param} || Trainable%: {100 * trainable_params / all_param}")

    def __call__(self, model, tokenizer):
        self.logging.info("Loading the LLM onto Lora Configuration....")
        model.gradient_checkpointing_enable()
        if self.use_lora:
            model              = prepare_model_for_kbit_training(model)
            model              = get_peft_model(model, self.lora_config)
        self.print_trainable_parameters(model)
        tokenizer.pad_token    = tokenizer.eos_token

        self.logging.info("Loading the Training and Dev Datasets....")
        train_dataset          = load_from_disk(self.path_dataset_train)
        eval_dataset           = load_from_disk(self.path_dataset_dev)
        train_dataset          = train_dataset.map(lambda samples: tokenizer(samples["text"]), batched=True)
        eval_dataset           = eval_dataset.map(lambda samples: tokenizer(samples["text"]), batched=True)
        
        self.logging.info("Preparing the Transformer Trainer....")
        trainer                = transformers.Trainer(
                                model=model,
                                train_dataset=train_dataset,##########################
                                eval_dataset=eval_dataset,
                                args=self.train_config,
                                data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False))
        model.config.use_cache = False

        self.logging.info(f"Finetuning the Backbone [{self.llm_backbone}] ....")
        trainer.train()
        tokenizer.save_pretrained(self.checkpoint_path)
        trainer.save_model()

        return model, trainer