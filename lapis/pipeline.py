import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
from peft import PeftModel, PeftConfig
import argparse
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
import json
from tqdm import tqdm
import openai
import os
from langchain_community.vectorstores import FAISS
import re
import pandas as pd
import argparse
from omegaconf import OmegaConf
import argparse
import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score


def true_or_false(x):
    if 'True' in x['results']:
        return True
    elif 'False' in x['results']:
        return False

    else:
        try:
            if '참' in x['results'].split('추론 결과')[1]:
                return True
            if '거짓' in x['results'].split('추론 결과')[1]:
                return False
        except:
            return np.nan

def bool_type_transform(x):
    return x['hypothesis_answer'].tolist()

def classification_scores(**kwargs):
    result_dict = dict()
    result_dict['f1-score']  = f1_score(**kwargs)
    result_dict['precision'] = precision_score(**kwargs)
    result_dict['recall']    = recall_score(**kwargs)
    result_dict['accuracy']  = accuracy_score(**kwargs)

    return result_dict
class LapisPipeline(object):
    def __init__(self, conf, logging):
        self.logging            = logging

        path_dataset            = os.path.join(conf.path.dataset, conf.dataprep.finetuning_dataset)
        self.path_dataset_test  = os.path.join(path_dataset, f"{conf.dataprep.instruction_method}_test_{conf.dataprep.subsample}.hf")
        self.path_dataset_train = os.path.join(path_dataset,
                                              f"{conf.dataprep.instruction_method}_train_{conf.dataprep.subsample}.hf")
        path_template           = os.path.join(conf.path.template, f"{conf.inference.template_method}.txt")

        if conf.inference.checkpoint_name is None:
            wandb_name              = f'{conf.wandb.project_name}_{conf.wandb.group_name}_{conf.wandb.session_name}/'
            self.path_checkpoint    = os.path.join(conf.path.checkpoint, wandb_name)
            self.path_result        = os.path.join(conf.path.result, wandb_name)
        else:
            self.path_checkpoint    = os.path.join(conf.path.checkpoint, conf.inference.checkpoint_name)
            self.path_result        = os.path.join(conf.path.result, conf.inference.checkpoint_name)


        self.logging.info(f"Loading Model Checkpoint from >>>>> {self.path_checkpoint}")
        if conf.finetune.lora.qlora:
            bnb_config              = BitsAndBytesConfig(
                                    load_in_4bit=True,
                                    bnb_4bit_use_double_quant=True,
                                    bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.bfloat16)
            self.model              = AutoModelForCausalLM.from_pretrained(
                                    self.path_checkpoint, 
                                    quantization_config=bnb_config, 
                                    device_map='cuda',
                                    local_files_only=True)
        else:
            print('no qlora, use full parameter')
            print('checkout this line in pipeline.py')
            self.model              = AutoModelForCausalLM.from_pretrained(
                        self.path_checkpoint,
                        device_map='cuda',
                        local_files_only=True)
        self.tokenizer          = AutoTokenizer.from_pretrained(self.path_checkpoint,local_files_only=True)
        self.tokenizer.pad_token= self.tokenizer.eos_token
        self.model.eval()


        self.logging.info(f"Loading Template Text File from >>>>> {path_template}")
        try:
            with open(path_template, 'r', encoding='cp949') as f:
                self.template = f.read()
        except:
            with open(path_template, 'r', encoding='utf-8') as f:
                self.template = f.read()


        self.logging.info(f"Loading Retrieval Modules >>>>> {conf.inference.retrieval.embedding_library}")
        self.logging.info(f"Loading Retrieval Modules >>>>> {conf.inference.retrieval.embedding_model}")
        self.logging.info(f"Loading Retrieval Modules >>>>> {conf.inference.retrieval.vector_library}")
        self.logging.info(f"Loading Retrieval Modules >>>>> {conf.inference.retrieval.vector_store}")
        os.environ['OPENAI_API_KEY'] = conf.openai.api_key
        if conf.inference.retrieval.embedding_library == 'openai':
            self.vector_embeddings         = OpenAIEmbeddings(model=conf.inference.retrieval.embedding_model)
        else:
            raise
        if conf.inference.retrieval.vector_library == 'faiss':
            self.vector_store              = FAISS.load_local(
                                            conf.inference.retrieval.vector_store,
                                            self.vector_embeddings,
                                            allow_dangerous_deserialization=True)
        elif conf.inference.retrieval.vector_library is None:
            self.vector_store              = None
        else:
            raise

        self.max_new_tokens = conf.inference.generation.max_new_tokens

        self.logging.info(f"The Results will be saved in >>>>> {self.path_result}")
        os.makedirs(self.path_result, exist_ok=True)
        OmegaConf.save(config=conf, f=os.path.join(self.path_result, 'config.yaml'))

    @torch.no_grad()
    def infer(self, hypothesis, premise):
        prompt  = PromptTemplate.from_template(self.template)
        inputs  = self.tokenizer(prompt.format(legal_hypothesis=text, premise=premise), return_tensors="pt").to('cuda')
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, eos_token_id=2)
        result  = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return result

    @torch.no_grad()
    def infer_without_premise(self, text):
        prompt  = PromptTemplate.from_template(self.template)
        inputs  = self.tokenizer(prompt.format(legal_hypothesis=text, premise=''), return_tensors="pt").to('cuda')
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, eos_token_id=2)
        result  = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return result

    @torch.no_grad()
    def infer_batch(self, batch):
        prompt  = PromptTemplate.from_template(self.template)
        inputs  = [prompt.format(legal_hypothesis=i,premise=j) for i,j in zip(batch['hypothesis'],batch['premise_retrieved'])]
        inputs  = self.tokenizer(inputs, return_tensors="pt", padding=True).to('cuda')
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, eos_token_id=2,
                                      do_sample=False,
                                      repetition_penalty=2.0)
        batch['results'] = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return batch

    @torch.no_grad()
    def infer_batch_without_premise(self, batch):
        prompt           = PromptTemplate.from_template(self.template)
        inputs           = [prompt.format(legal_hypothesis=i, premise='') for i in batch['hypothesis']]
        inputs           = self.tokenizer(inputs, return_tensors="pt", padding=True).to('cuda')
        outputs          = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, eos_token_id=2)
        batch['results'] = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return batch

    @torch.no_grad()
    def add_premise_to_batch(self, batch):
        batch['premise_retrieved'] = []
        if self.vector_store:
            for hypothesis in batch['hypothesis']:
                hypothesis_embedding          = self.vector_embeddings.embed_query(hypothesis)
                docs                          = self.vector_store.similarity_search_by_vector(
                                                hypothesis_embedding,
                                                k=5,
                                                fetch_k=10)
                premise = \
                    '\n'+'premise 1:'+re.sub(': \d{,6}\nsummary:', '', docs[0].page_content) + '\n' + \
                    'premise 2:'+re.sub(': \d{,6}\nsummary:', '', docs[1].page_content) + '\n' + \
                    'premise 3:'+re.sub(': \d{,6}\nsummary:', '', docs[2].page_content) + '\n' + \
                    'premise 4:'+re.sub(': \d{,6}\nsummary:', '', docs[3].page_content) + '\n' + \
                    'premise 5:'+re.sub(': \d{,6}\nsummary:', '', docs[4].page_content)

                batch['premise_retrieved'].append(premise)

        return batch


    def __call__(self, **kwargs):
        try:
            hypothesis_id             = kwargs['hypothesis_id']
        except:
            hypothesis_id             = 'ID-UNKNOWN'
        year                          = kwargs['year']
        subject                       = kwargs['subject']
        ground_truth                  = kwargs['hypothesis_answer']
        hypothesis                    = kwargs['hypothesis']
        premise                       = ''

        if self.vector_store:
            hypothesis_embedding          = self.vector_embeddings.embed_query(hypothesis)
            docs                          = self.vector_store.similarity_search_by_vector(
                                            hypothesis_embedding,
                                            k=5,
                                            fetch_k=10)
            premise = \
                re.sub(': \d{,6}\nsummary:', '', docs[0].page_content) + ' ' + \
                re.sub(': \d{,6}\nsummary:', '', docs[1].page_content) + ' ' + \
                re.sub(': \d{,6}\nsummary:', '', docs[2].page_content) + ' ' + \
                re.sub(': \d{,6}\nsummary:', '', docs[3].page_content) + ' ' + \
                re.sub(': \d{,6}\nsummary:', '', docs[4].page_content)
        
        try:
            if premise == '':
                output = self.infer_without_premise(hypothesis)
                return [hypothesis_id, year, subject, ground_truth, hypothesis, output]
            else:
                output = self.infer(hypothesis, premise)
                return [hypothesis_id, year, subject, ground_truth, hypothesis, output]
        except Exception as e:
            self.logging.info(f"ERROR: {hypothesis_id} >>>>> {e}")