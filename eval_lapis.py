import os
from os import walk
import pdb
from lapis.dataset import RawDataset
from lapis.pipeline import LapisPipeline, true_or_false, classification_scores, bool_type_transform
import logging
from omegaconf import OmegaConf
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
import gc
import argparse
import json
import numpy as np
import random

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.set_num_threads(1)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO, 
                    datefmt='%Y-%m-%d %H:%M:%S')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--omegaconf',   '-oc', type=str, default='lapis_yanolja+EEVE-Korean-10.8B-v1.0_True')
    parser.add_argument('--random_seed', '-rs', type=int, default=8888)
    args   = parser.parse_args()
    conf   = OmegaConf.load('./settings.yaml')[args.omegaconf]
    setup_seed(args.random_seed)

    PREP_PATH = os.path.join(conf.path.dataset, conf.dataprep.finetuning_dataset)
    logging.info(f"STEP [1] : Preparing the instruction fine-tuning dataset >>>>> {PREP_PATH}")
    if not os.path.isdir(f"{PREP_PATH}/{conf.dataprep.instruction_method}_train.hf"):
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
        elif conf.dataprep.instruction_method == '6s_solution':
            raw_dataset.make_6s_solution_instruction_dataset()

        else:
            raise


    logging.info(f"STEP [2] : Preparing the Lapis Pipeline to be evaluated >>>>> {conf.finetune.llm_backbone}")
    pipeline        = LapisPipeline(conf, logging)
    test_dataset    = load_from_disk(pipeline.path_dataset_test)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    print(len(test_dataset))


    logging.info(f"STEP [3] : Running Inference on the Lapis Pipeline >>>>> {conf.finetune.llm_backbone}")
    START_FLAG  = False
    num_batches = len(test_dataloader)
    for idx, batch in enumerate(test_dataloader):

        if idx == conf.inference.start_batch:
            START_FLAG = True
        if START_FLAG:
            logging.info("Retrieving Premises...")
            batch = pipeline.add_premise_to_batch(batch)
            logging.info("Running Inference...")
            batch = pipeline.infer_batch(batch)
            #batch = pipeline.infer_batch_without_premise(batch)
            SAVE_PATH = os.path.join(pipeline.path_result, f"{conf.inference.template_method}_inference_results_{idx:06d}_{num_batches:06d}.csv" )
            pd.DataFrame.from_dict(batch).to_csv(SAVE_PATH, index='hypothesis_id', encoding='utf-8-sig')
            logging.info(SAVE_PATH)
            print("")
        gc.collect()
        torch.cuda.empty_cache()

    logging.info(f"STEP [4] : Evaluating the Inference Results >>>>> {pipeline.path_result}")
    f, list_dataframes = [], []
    for (dirpath, dirnames, filenames) in os.walk(pipeline.path_result):
        f.extend(filenames)
    for filename in filenames:
        if f'_{num_batches:06d}.csv' in filename:
            list_dataframes.append(pd.read_csv(os.path.join(pipeline.path_result, filename), encoding='utf-8-sig'))
    df                       = pd.concat(list_dataframes)
    df['hypothesis_predict'] = df.apply(lambda x: true_or_false(x), axis=1)

    df['hypothesis_answer'] = df.apply(lambda x: bool_type_transform(x), axis=1)
    result_dict              = classification_scores(y_true=df['hypothesis_answer'],y_pred=df['hypothesis_predict'])
    logging.info(result_dict)
    SCORES_PATH              = os.path.join(pipeline.path_result, f'{conf.inference.template_method}_inference_scores.json')
    RESULTS_PATH             = os.path.join(pipeline.path_result, f'{conf.inference.template_method}_inference_results.csv')
    with open(SCORES_PATH, 'w') as fp:
        json.dump(result_dict, fp)
    df.to_csv(RESULTS_PATH, index='hypothesis_id', encoding='utf-8-sig')