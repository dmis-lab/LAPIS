from omegaconf import OmegaConf
import itertools 
import os
import multiprocessing as mp
import argparse

from copy import deepcopy


parser = argparse.ArgumentParser()

parser.add_argument('--project_name', '-pn', type=str, default='CIKM_Lapis_2024')
parser.add_argument('--group_name',   '-gn', type=str)
parser.add_argument('--ncpu',         '-nc', type=int, default=1)
parser.add_argument('--multi_gpu',    '-mg', type=str, default='0,1')
parser.add_argument('--test_run',     '-tr', default=False, action='store_true')
config = parser.parse_args()

def run_experiment():

    return

def make_yaml_configs(conf, file_path="./settings.yaml"):
    new_conf               = dict()

    KOREAN_LLMS = [
                    # 'yanolja/EEVE-Korean-10.8B-v1.0',
                    # 'yanolja/KoSOLAR-10.7B-v0.2',
                    'beomi/gemma-mling-7b',
                    'beomi/gemma-ko-2b',
                    'beomi/gemma-ko-7b',
                    'beomi/OPEN-SOLAR-KO-10.7B',
                    'beomi/llama-2-ko-7b',
                    'beomi/Yi-Ko-6B',
                    'beomi/KoAlpaca-Polyglot-5.8B',
                    'beomi/KoAlpaca-Polyglot-12.8B',
                    'beomi/KoAlpaca-llama-1-7b'
                   ]
    QLORA       = [True]

    possible_args = itertools.product(KOREAN_LLMS, QLORA)
    for x in possible_args:
        x0  = x[0].replace('/','+')
        key = f'lapis_{x0}_{x[1]}'
        new_conf[key]                                 = deepcopy(dict(conf['search']))
        new_conf[key]['wandb']['project_name']        = config.project_name
        new_conf[key]['wandb']['group_name']          = config.group_name
        new_conf[key]['wandb']['session_name']        = key
        new_conf[key]['finetune']['llm_backbone']     = x[0]
        new_conf[key]['finetune']['lora']['qlora']    = x[1]

        if config.test_run:
            new_conf[key]['dataprep']['subsample']    = 0.001
            new_conf[key]['wandb']['group_name']      = 'testrun'
            new_conf[key]['wandb']['session_name']    = key+'_testrun'

    new_conf = OmegaConf.create(new_conf)
    OmegaConf.save(config=new_conf, f='settings.yaml')

    return list(new_conf.keys())

def single_process(x):
    print(x)
    os.system(x)

def multi_process(list_omegaconf_keys: list):
    print(len(list_omegaconf_keys))
    list_python_scripts = [f"CUDA_VISIBLE_DEVICES={config.multi_gpu} python -W ignore train_lapis.py -oc {x}" for i, x in enumerate(list_omegaconf_keys)]

    with mp.Pool(config.ncpu) as pool:
        r = pool.map_async(single_process, list_python_scripts)
        r.wait()
        pool.close()
        pool.join()

if __name__ == "__main__":
    conf = OmegaConf.load(f'./lapis/default_settings.yaml')

    multi_process(make_yaml_configs(conf))





