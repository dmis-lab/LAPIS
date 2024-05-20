import os
import pdb
from lapis.dataset import RawDataset
from lapis.trainer import LapisTrainer
import logging
from omegaconf import OmegaConf
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from env import *
import setproctitle
import argparse

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.set_num_threads(1)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO, 
                    datefmt='%Y-%m-%d %H:%M:%S')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--omegaconf', '-oc', type=str, default='dev')
    args   = parser.parse_args()
    conf   = OmegaConf.load('./settings.yaml')[args.omegaconf]

    PREP_PATH = os.path.join(conf.path.dataset, conf.dataprep.finetuning_dataset)
    logging.info(f"STEP [1] : Preparing the instruction fine-tuning dataset >>>>> {PREP_PATH}")
    if not os.path.isdir(f"{PREP_PATH}/{conf.dataprep.instruction_method}_train_{conf.dataprep.subsample}.hf"):
        raw_dataset = RawDataset(conf, logging)
        if conf.dataprep.instruction_method == 'explain':
            raw_dataset.make_expert_explain_instruction_dataset()
        elif conf.dataprep.instruction_method == 'correct':
            raw_dataset.make_expert_correct_instruction_dataset()
        elif conf.dataprep.instruction_method == 'correct_explain':
            raw_dataset.make_correct_explain_instruction_dataset()
        elif conf.dataprep.instruction_method == 'only_3s_setting':
            raw_dataset.make_only_3s_setting_instruction_dataset()
        elif conf.dataprep.instruction_method == 'expert_curation_only':
            raw_dataset.make_expert_curation_only_instruction_dataset()
        elif conf.dataprep.instruction_method == '6s_rationales':
            raw_dataset.make_6s_rationales_instruction_dataset()
        elif conf.dataprep.instruction_method == 'simple':
            raw_dataset.make_simple_finetuning_instruction_dataset()
        elif conf.dataprep.instruction_method == '6s_solution':
            raw_dataset.make_6s_solution_instruction_dataset()

        else:
            raise


    logging.info(f"STEP [2] : Preparing the LLM to be fine-tuned >>>>> {conf.finetune.llm_backbone}")
    setproctitle.setproctitle(f"finetuning_{conf.finetune.llm_backbone}")
    tokenizer  = AutoTokenizer.from_pretrained(conf.finetune.llm_backbone)
    if conf.finetune.lora.qlora and conf.finetune.lora.enabled:
        bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16)
        model      = AutoModelForCausalLM.from_pretrained(
                        conf.finetune.llm_backbone, 
                        quantization_config=bnb_config, 
                        device_map="auto")
    else:
        model      = AutoModelForCausalLM.from_pretrained(
                conf.finetune.llm_backbone,
                device_map="auto")


    logging.info(f"STEP [3] : Initiating the fine-tuning phase >>>>> {conf.finetune.llm_backbone}")
    trainer    = LapisTrainer(conf, logging)
    trainer(model, tokenizer)