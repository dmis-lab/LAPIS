import os
import json
import pandas as pd
from datasets import Dataset, concatenate_datasets


class RawDataset(object):
    def __init__(self, conf, logging):
        self.path_rawdata = os.path.join(conf.path.dataset, conf.dataprep.raw_dataset)
        self.path_ftdata  = os.path.join(conf.path.dataset, conf.dataprep.finetuning_dataset)
        self.logging  = logging

        self.years_train = conf.dataprep.years_train
        self.years_dev   = conf.dataprep.years_dev
        self.years_test  = conf.dataprep.years_test
        self.subsample   = conf.dataprep.subsample


        if not isinstance(self.years_train, list):
            self.years_train = [i for i in range(2013,2024) if i not in self.years_dev + self.years_test]

        self.logging.info(f"[TRAIN]    Dataset Partition Years:         {self.years_train}")
        self.logging.info(f"[DEV]      Dataset Partition Years:         {self.years_dev}")
        self.logging.info(f"[TEST]     Dataset Partition Years:         {self.years_test}")
        self.logging.info(f"[SUBSAMPLE]                       :         {self.subsample}")

        self.ft_dataset = conf.dataprep.finetuning_dataset


    def make_expert_explain_instruction_dataset(self):
        COLUMNS = ['hypothesis', 'output', 'year', 'hypothesis_id', 'subject', 'hypothesis_answer']
        """
        2013~2020 년도에 해당하는 가설데이터에 대해서만 hypothesis, expert explain 정보를 추출해서 instruction formatted data로 변형함
        """
        with open(self.path_rawdata+".json", 'r') as f:
            data = f.read()
        raw_dataset = json.loads(data)

        data_instances = dict(train=[],dev=[],test=[])

        for key in raw_dataset:
            year = int(raw_dataset[key]['year'])
            key_split = None
            dict_data = dict()
            dict_data['hypothesis']        = raw_dataset[key]['hypothesis']
            dict_data['year']              = year
            dict_data['hypothesis_id']     = raw_dataset[key]['hypothesis_id']
            dict_data['subject']           = raw_dataset[key]['subject']
            dict_data['hypothesis_answer'] = raw_dataset[key]['hypothesis_answer']

            if year in self.years_train:
                key_split = 'train'
            elif year in self.years_dev:
                key_split = 'dev'
            elif year in self.years_test:
                key_split = 'test'
            else:
                raise

            if raw_dataset[key]['expert_explain'] != 'nan' and raw_dataset[key]['expert_explain'] != 'None':
                dict_data['output'] = raw_dataset[key]['expert_explain']
                data_instances[key_split].append(dict_data)

        for key_split in ['train', 'dev', 'test']:

            data = Dataset.from_pandas(pd.DataFrame(data=data_instances[key_split],
                                                    columns=COLUMNS).sample(frac=self.subsample))
            data = Dataset.from_pandas(pd.DataFrame(data=data_instances[key_split],
                                                    columns=COLUMNS).sample(frac=self.subsample))
            data = data.map(
                lambda x: {'text':
    f"""당신은 범죄 수사 전문가로, 주어진 법률 가설의 참, 거짓을 판단할 수 있습니다. 당신의 임무는 주어진 법률 가설을 검토하여, 법률 가설이 참인지 거짓인지를 판단하는 것입니다. 이때, 당신이 보유한 법률 전문 지식과 추론 능력을 이용해 법률 가설의 참 거짓을 판단하십시오. 법률 가설의 참 거짓을 판단할 때는 반드시 적절한 법리, 판결문, 논리적 구성 요소에 따라 진행하십시오.
    ---

    항상 다음과 같은 형식을 따라야 합니다.

    법률 가설 : "법률, 규칙, 법리, 범죄 상황 등과 관련한 주장"

    추론 결과 : "참 또는 거짓. 법률 가설의 판단. "

    ---

    법률 가설 : {x['hypothesis']}
    추론 결과 : {x['output']}<|endoftext|>""" })

            data.save_to_disk(f"{self.path_ftdata}/explain_{key_split}_{self.subsample}.hf")


    def make_expert_correct_instruction_dataset(self):
        COLUMNS = ['hypothesis', 'premise', 'output', 'year', 'hypothesis_id', 'subject', 'hypothesis_answer'] # 희두 : 'expert_correction' 컬럼 추가됨(CI_hypothesis_v1.json)
        """
        2013~2020 년도에 해당하는 가설데이터에 대해서만 hypothesis, expert explain 정보를 추출해서 instruction formatted data로 변형함
        """
        with open(self.path_rawdata+".json", 'r') as f:
            data = f.read()
        raw_dataset = json.loads(data)

        data_instances = dict(train=[],dev=[],test=[])
        data_instances_test = dict(train=[], dev=[], test=[])
        for key in raw_dataset:
            year = int(raw_dataset[key]['year'])
            key_split = None
            gpt_rationales = raw_dataset[key]['gpt_rationales']

            if year in self.years_train:
                key_split = 'train'
            elif year in self.years_dev:
                key_split = 'dev'
            elif year in self.years_test:
                key_split = 'test'
            else:
                raise

            for n in [f'gpt4_rationale_{i}' for i in range(1,7)]:
                dict_data = dict()
                dict_data['hypothesis']        = raw_dataset[key]['hypothesis']
                dict_data['premise']           = raw_dataset[key]['premise']
                dict_data['output']            = raw_dataset[key]['gpt_rationales'][n]['text']
                # dict_data['output_corrected']  = None
                dict_data['year']              = year
                dict_data['hypothesis_id']     = raw_dataset[key]['hypothesis_id'] + '_' + n
                dict_data['subject']           = raw_dataset[key]['subject']
                dict_data['hypothesis_answer'] = raw_dataset[key]['hypothesis_answer']
                if raw_dataset[key]['gpt_rationales'][n]['gpt4_is_correct'] == 'correct':
                    data_instances[key_split].append(dict_data)

            try:
                dict_data = dict()
                dict_data['hypothesis']        = raw_dataset[key]['hypothesis']
                dict_data['premise']           = raw_dataset[key]['premise']
                dict_data['output']            = raw_dataset[key]['expert_correction']
                # dict_data['output_corrected']  = None
                dict_data['year']              = year
                dict_data['hypothesis_id']     = raw_dataset[key]['hypothesis_id'] + '_expert_corrected' 
                dict_data['subject']           = raw_dataset[key]['subject']
                dict_data['hypothesis_answer'] = raw_dataset[key]['hypothesis_answer']
                data_instances[key_split].append(dict_data)
                data_instances_test[key_split].append(dict_data)
            except:
                pass

        for key_split in ['train', 'dev', 'test']:
            if key_split == 'train' or key_split == 'dev':
                data = Dataset.from_pandas(pd.DataFrame(data=data_instances[key_split],
                                                        columns=COLUMNS).sample(frac=self.subsample))
                data = data.map(
                    lambda x: {'text':
       f"""당신은 범죄 수사 전문가로, 주어진 법률 가설의 참, 거짓을 판단할 수 있습니다. 당신의 임무는 주어진 법률 가설과 전제를 검토하여, 법률 가설이 참인지 거짓인지를 판단하는 것입니다. 이때, 다음과 같은 과정을 거쳐서 판단해야 합니다. 먼저, 근거로 제시된 전제들을 자세히 읽어보어야 합니다. 이어서, 제시된 전제들이 법률 가설 판단에 충분한 도움이 된다면, 그 전제의 내용을 이용해 법률 가설의 참 거짓을 판단하십시오. 만약 제시된 전제들이 도움이 되지 않는다면, 당신이 보유한 법률 전문 지식과 추론 능력을 이용해 법률 가설의 참 거짓을 판단하십시오. 법률 가설의 참 거짓을 판단할 때는 반드시 적절한 법리, 판결문, 논리적 구성 요소에 따라 판단하십시오.
        ---
    
        항상 다음과 같은 형식을 따라야 합니다.
    
        법률 가설 : "법률, 규칙, 법리, 범죄 상황 등과 관련한 주장"
    
        전제 : "5개의 전제 사실"
        전제 i: "i번째 전제 사실, 판결문, 법률 조항, 범죄 수사학 교과서 등의 정보로써 법률 가설 판단에 도움이 될 것으로 예상됨"
    
        추론: "법률 가설의 판단을 위한 추론 과정"
    
        답변: "법률 가설의 판단. 참 또는 거짓."
    
        ---
    
        법률 가설: {x['hypothesis']}
    
        전제: {x['premise']}
    
        추론 결과:{x['output']}<|endoftext|>"""})

                data.save_to_disk(f"{self.path_ftdata}/correct_{key_split}_{self.subsample}.hf")

            elif key_split == 'test':
                data = Dataset.from_pandas(pd.DataFrame(data=data_instances_test[key_split],
                                                        columns=COLUMNS).sample(frac=self.subsample))
                data.save_to_disk(f"{self.path_ftdata}/correct_{key_split}_{self.subsample}.hf")

    def make_correct_explain_instruction_dataset(self):
        COLUMNS = ['hypothesis', 'premise', 'output', 'year', 'hypothesis_id', 'subject',
                   'hypothesis_answer']
        """
        2013~2020 년도에 해당하는 가설데이터에 대해서만 hypothesis, expert explain 정보를 추출해서 instruction formatted data로 변형함
        """
        with open(self.path_rawdata + ".json", 'r') as f:
            data = f.read()
        raw_dataset = json.loads(data)

        data_instances = dict(train=[], dev=[], test=[])
        data_instances_test = dict(train=[], dev=[], test=[])
        data_instances_explain = dict(train=[], dev=[], test=[])

        for key in raw_dataset:
            year = int(raw_dataset[key]['year'])
            key_split = None
            gpt_rationales = raw_dataset[key]['gpt_rationales']

            if year in self.years_train:
                key_split = 'train'
            elif year in self.years_dev:
                key_split = 'dev'
            elif year in self.years_test:
                key_split = 'test'
            else:
                raise

            for n in [f'gpt4_rationale_{i}' for i in range(1, 7)]:
                dict_data = dict()
                dict_data['hypothesis'] = raw_dataset[key]['hypothesis']
                dict_data['premise'] = raw_dataset[key]['premise']
                dict_data['output'] = raw_dataset[key]['gpt_rationales'][n]['text']
                # dict_data['output_corrected']  = None
                dict_data['year'] = year
                dict_data['hypothesis_id'] = raw_dataset[key]['hypothesis_id'] + '_' + n
                dict_data['subject'] = raw_dataset[key]['subject']
                dict_data['hypothesis_answer'] = raw_dataset[key]['hypothesis_answer']
                if raw_dataset[key]['gpt_rationales'][n]['gpt4_is_correct'] == 'correct':
                    data_instances[key_split].append(dict_data)

            try:
                dict_data = dict()
                dict_data['hypothesis'] = raw_dataset[key]['hypothesis']
                dict_data['premise'] = raw_dataset[key]['premise']
                dict_data['output'] = raw_dataset[key]['expert_correction']
                # dict_data['output_corrected']  = None
                dict_data['year'] = year
                dict_data['hypothesis_id'] = raw_dataset[key]['hypothesis_id'] + '_expert_corrected'
                dict_data['subject'] = raw_dataset[key]['subject']
                dict_data['hypothesis_answer'] = raw_dataset[key]['hypothesis_answer']
                data_instances[key_split].append(dict_data)
                data_instances_test[key_split].append(dict_data)
            except:
                pass

            if raw_dataset[key]['expert_explain'] != 'nan' and raw_dataset[key]['expert_explain'] != 'None':
                dict_data = dict()
                dict_data['hypothesis'] = raw_dataset[key]['hypothesis']
                dict_data['premise'] = raw_dataset[key]['premise']
                dict_data['output'] = raw_dataset[key]['expert_explain']
                dict_data['year'] = year
                dict_data['hypothesis_id'] = raw_dataset[key]['hypothesis_id'] + '_'  + '_expert_explain'
                dict_data['subject'] = raw_dataset[key]['subject']
                dict_data['hypothesis_answer'] = raw_dataset[key]['hypothesis_answer']
                data_instances_explain[key_split].append(dict_data)

        for key_split in ['train', 'dev', 'test']:
            if key_split == 'train' or key_split == 'dev':
                data = Dataset.from_pandas(pd.DataFrame(data=data_instances[key_split],
                                                        columns=COLUMNS).sample(frac=self.subsample))
                data_explain = Dataset.from_pandas(pd.DataFrame(data=data_instances_explain[key_split],
                                                        columns=COLUMNS).sample(frac=self.subsample))

                data = data.map(
                    lambda x: {'text':
                                   f"""당신은 범죄 수사 전문가로, 주어진 법률 가설의 참, 거짓을 판단할 수 있습니다. 당신의 임무는 주어진 법률 가설과 전제를 검토하여, 법률 가설이 참인지 거짓인지를 판단하는 것입니다. 이때, 다음과 같은 과정을 거쳐서 판단해야 합니다. 먼저, 근거로 제시된 전제들을 자세히 읽어보어야 합니다. 이어서, 제시된 전제들이 법률 가설 판단에 충분한 도움이 된다면, 그 전제의 내용을 이용해 법률 가설의 참 거짓을 판단하십시오. 만약 제시된 전제들이 도움이 되지 않는다면, 당신이 보유한 법률 전문 지식과 추론 능력을 이용해 법률 가설의 참 거짓을 판단하십시오. 법률 가설의 참 거짓을 판단할 때는 반드시 적절한 법리, 판결문, 논리적 구성 요소에 따라 판단하십시오.
         ---
    
         항상 다음과 같은 형식을 따라야 합니다.
    
         법률 가설 : "법률, 규칙, 법리, 범죄 상황 등과 관련한 주장"
    
         전제 : "5개의 전제 사실"
         전제 i: "i번째 전제 사실, 판결문, 법률 조항, 범죄 수사학 교과서 등의 정보로써 법률 가설 판단에 도움이 될 것으로 예상됨"
    
         추론: "법률 가설의 판단을 위한 추론 과정"
    
         답변: "법률 가설의 판단. 참 또는 거짓."
    
         ---
    
         법률 가설: {x['hypothesis']}
    
         전제: {x['premise']}
    
         추론 결과:{x['output']}<|endoftext|>"""})

                data_explain = data_explain.map(
                    lambda x: {'text':
                                   f"""당신은 범죄 수사 전문가로, 주어진 법률 가설의 참, 거짓을 판단할 수 있습니다. 당신의 임무는 주어진 법률 가설을 검토하여, 법률 가설이 참인지 거짓인지를 판단하는 것입니다. 이때, 당신이 보유한 법률 전문 지식과 추론 능력을 이용해 법률 가설의 참 거짓을 판단하십시오. 법률 가설의 참 거짓을 판단할 때는 반드시 적절한 법리, 판결문, 논리적 구성 요소에 따라 진행하십시오.
                ---
    
                항상 다음과 같은 형식을 따라야 합니다.
    
                법률 가설 : "법률, 규칙, 법리, 범죄 상황 등과 관련한 주장"
    
                추론: "법률 가설의 판단을 위한 추론 과정"
    
                답변: "법률 가설의 판단. 참 또는 거짓."
    
                ---
    
                법률 가설 : {x['hypothesis']}
                추론 결과 : {x['output']}<|endoftext|>"""})

                dataset_cc = concatenate_datasets([data, data_explain])

                dataset_cc.save_to_disk(f"{self.path_ftdata}/correct_explain_{key_split}_{self.subsample}.hf")

            elif key_split == 'test':
                data_test = Dataset.from_pandas(pd.DataFrame(data=data_instances_test[key_split],
                                                        columns=COLUMNS).sample(frac=self.subsample))

                data_test.save_to_disk(f"{self.path_ftdata}/correct_explain_{key_split}_{self.subsample}.hf")

    def make_only_3s_setting_instruction_dataset(self):
        COLUMNS = ['hypothesis', 'premise', 'output', 'year', 'hypothesis_id', 'subject',
                   'hypothesis_answer']
        """
        2013~2020 년도에 해당하는 가설데이터에 대해서만 hypothesis, expert explain 정보를 추출해서 instruction formatted data로 변형함
        """
        with open(self.path_rawdata + ".json", 'r') as f:
            data = f.read()
        raw_dataset = json.loads(data)

        data_instances = dict(train=[], dev=[], test=[])
        data_instances_test = dict(train=[], dev=[], test=[])

        for key in raw_dataset:
            year = int(raw_dataset[key]['year'])
            key_split = None

            if year in self.years_train:
                key_split = 'train'
            elif year in self.years_dev:
                key_split = 'dev'
            elif year in self.years_test:
                key_split = 'test'
            else:
                raise


            dict_data = dict()
            dict_data['hypothesis'] = raw_dataset[key]['hypothesis']
            dict_data['premise'] = raw_dataset[key]['premise']
            dict_data['output'] = raw_dataset[key]['gpt_rationales']['gpt4_rationale_6']['text']
            # dict_data['output_corrected']  = None
            dict_data['year'] = year
            dict_data['hypothesis_id'] = raw_dataset[key]['hypothesis_id']
            dict_data['subject'] = raw_dataset[key]['subject']
            dict_data['hypothesis_answer'] = raw_dataset[key]['hypothesis_answer']
            data_instances[key_split].append(dict_data)

        for key_split in ['train', 'dev', 'test']:
            data = Dataset.from_pandas(pd.DataFrame(data=data_instances[key_split],
                                                    columns=COLUMNS).sample(frac=self.subsample))

            data = data.map(
                lambda x: {'text':
                               f"""당신은 범죄 수사 전문가로, 주어진 법률 가설의 참, 거짓을 판단할 수 있습니다. 당신의 임무는 주어진 법률 가설과 전제를 검토하여, 법률 가설이 참인지 거짓인지를 판단하는 것입니다. 이때, 다음과 같은 과정을 거쳐서 판단해야 합니다. 먼저, 근거로 제시된 전제들을 자세히 읽어보어야 합니다. 이어서, 제시된 전제들이 법률 가설 판단에 충분한 도움이 된다면, 그 전제의 내용을 이용해 법률 가설의 참 거짓을 판단하십시오. 만약 제시된 전제들이 도움이 되지 않는다면, 당신이 보유한 법률 전문 지식과 추론 능력을 이용해 법률 가설의 참 거짓을 판단하십시오. 법률 가설의 참 거짓을 판단할 때는 반드시 적절한 법리, 판결문, 논리적 구성 요소에 따라 판단하십시오.
     ---

     항상 다음과 같은 형식을 따라야 합니다.

     법률 가설 : "법률, 규칙, 법리, 범죄 상황 등과 관련한 주장"

     전제 : "5개의 전제 사실"
     전제 i: "i번째 전제 사실, 판결문, 법률 조항, 범죄 수사학 교과서 등의 정보로써 법률 가설 판단에 도움이 될 것으로 예상됨"

     추론: "법률 가설의 판단을 위한 추론 과정"

     답변: "법률 가설의 판단. 참 또는 거짓."

     ---

     법률 가설: {x['hypothesis']}

     전제: {x['premise']}

     추론 결과:{x['output']}<|endoftext|>"""})

            data.save_to_disk(f"{self.path_ftdata}/only_3s_setting_{key_split}_{self.subsample}.hf")


    def make_expert_curation_only_instruction_dataset(self):
        COLUMNS = ['hypothesis', 'premise', 'output', 'year', 'hypothesis_id', 'subject',
                   'hypothesis_answer']
        """
        2013~2020 년도에 해당하는 가설데이터에 대해서만 hypothesis, expert explain 정보를 추출해서 instruction formatted data로 변형함
        """
        with open(self.path_rawdata + ".json", 'r') as f:
            data = f.read()
        raw_dataset = json.loads(data)

        data_instances = dict(train=[], dev=[], test=[])

        for key in raw_dataset:
            year = int(raw_dataset[key]['year'])
            key_split = None

            if year in self.years_train:
                key_split = 'train'
            elif year in self.years_dev:
                key_split = 'dev'
            elif year in self.years_test:
                key_split = 'test'
            else:
                raise


            dict_data = dict()
            dict_data['hypothesis'] = raw_dataset[key]['hypothesis']
            dict_data['premise'] = raw_dataset[key]['premise']
            if raw_dataset[key]['expert_correction'] != 'nan' and raw_dataset[key]['expert_correction'] != 'None':
                dict_data['output'] = raw_dataset[key]['expert_correction']
            else:
                dict_data['output'] = raw_dataset[key]['gpt_rationales']['gpt4_rationale_6']['text']
            # dict_data['output_corrected']  = None
            dict_data['year'] = year
            dict_data['hypothesis_id'] = raw_dataset[key]['hypothesis_id']
            dict_data['subject'] = raw_dataset[key]['subject']
            dict_data['hypothesis_answer'] = raw_dataset[key]['hypothesis_answer']
            data_instances[key_split].append(dict_data)

        for key_split in ['train', 'dev', 'test']:
            data = Dataset.from_pandas(pd.DataFrame(data=data_instances[key_split],
                                                    columns=COLUMNS).sample(frac=self.subsample))

            data = data.map(
                lambda x: {'text':
                               f"""당신은 범죄 수사 전문가로, 주어진 법률 가설의 참, 거짓을 판단할 수 있습니다. 당신의 임무는 주어진 법률 가설과 전제를 검토하여, 법률 가설이 참인지 거짓인지를 판단하는 것입니다. 이때, 다음과 같은 과정을 거쳐서 판단해야 합니다. 먼저, 근거로 제시된 전제들을 자세히 읽어보어야 합니다. 이어서, 제시된 전제들이 법률 가설 판단에 충분한 도움이 된다면, 그 전제의 내용을 이용해 법률 가설의 참 거짓을 판단하십시오. 만약 제시된 전제들이 도움이 되지 않는다면, 당신이 보유한 법률 전문 지식과 추론 능력을 이용해 법률 가설의 참 거짓을 판단하십시오. 법률 가설의 참 거짓을 판단할 때는 반드시 적절한 법리, 판결문, 논리적 구성 요소에 따라 판단하십시오.
     ---

     항상 다음과 같은 형식을 따라야 합니다.

     법률 가설 : "법률, 규칙, 법리, 범죄 상황 등과 관련한 주장"

     전제 : "5개의 전제 사실"
     전제 i: "i번째 전제 사실, 판결문, 법률 조항, 범죄 수사학 교과서 등의 정보로써 법률 가설 판단에 도움이 될 것으로 예상됨"

     추론 결과: "참 또는 거짓. 법률 가설의 판단을 위한 추론 과정"

     ---

     법률 가설: {x['hypothesis']}

     전제: {x['premise']}

     추론 결과:{x['output']}<|endoftext|>"""})

            data.save_to_disk(f"{self.path_ftdata}/expert_curation_only_{key_split}_{self.subsample}.hf")

    def make_6s_rationales_instruction_dataset(self):
        COLUMNS = ['hypothesis', 'premise', 'output', 'year', 'hypothesis_id', 'subject', 'hypothesis_answer'] # 희두 : 'expert_correction' 컬럼 추가됨(CI_hypothesis_v1.json)
        """
        2013~2020 년도에 해당하는 가설데이터에 대해서만 hypothesis, expert explain 정보를 추출해서 instruction formatted data로 변형함
        """
        with open(self.path_rawdata+".json", 'r') as f:
            data = f.read()
        raw_dataset = json.loads(data)

        data_instances = dict(train=[],dev=[],test=[])
        data_instance_test = dict(train=[], dev=[], test=[])

        for key in raw_dataset:
            year = int(raw_dataset[key]['year'])
            key_split = None
            gpt_rationales = raw_dataset[key]['gpt_rationales']

            if year in self.years_train:
                key_split = 'train'
            elif year in self.years_dev:
                key_split = 'dev'
            elif year in self.years_test:
                key_split = 'test'
            else:
                raise

            for n in [f'gpt4_rationale_{i}' for i in range(1,7)]:
                dict_data = dict()
                dict_data['hypothesis']        = raw_dataset[key]['hypothesis']
                dict_data['premise']           = raw_dataset[key]['premise']
                dict_data['output']            = raw_dataset[key]['gpt_rationales'][n]['text']
                # dict_data['output_corrected']  = None
                dict_data['year']              = year
                dict_data['hypothesis_id']     = raw_dataset[key]['hypothesis_id'] + '_' + n
                dict_data['subject']           = raw_dataset[key]['subject']
                dict_data['hypothesis_answer'] = raw_dataset[key]['hypothesis_answer']
                if raw_dataset[key]['gpt_rationales'][n]['gpt4_is_correct'] == 'correct':
                    data_instances[key_split].append(dict_data)
            if year in self.years_test:
                dict_data = dict()
                dict_data['hypothesis']        = raw_dataset[key]['hypothesis']
                dict_data['premise']           = raw_dataset[key]['premise']
                dict_data['output']            = raw_dataset[key]['gpt_rationales']['gpt4_rationale_6']['text']
                # dict_data['output_corrected']  = None
                dict_data['year']              = year
                dict_data['hypothesis_id']     = raw_dataset[key]['hypothesis_id']
                dict_data['subject']           = raw_dataset[key]['subject']
                dict_data['hypothesis_answer'] = raw_dataset[key]['hypothesis_answer']
                data_instance_test[key_split].append(dict_data) #희두 추가


        for key_split in ['train', 'dev', 'test']:
            if key_split == 'train' or key_split == 'dev':
                data = Dataset.from_pandas(pd.DataFrame(data=data_instances[key_split],
                                                        columns=COLUMNS).sample(frac=self.subsample))
                data = data.map(
                    lambda x: {'text':
       f"""당신은 범죄 수사 전문가로, 주어진 법률 가설의 참, 거짓을 판단할 수 있습니다. 당신의 임무는 주어진 법률 가설과 전제를 검토하여, 법률 가설이 참인지 거짓인지를 판단하는 것입니다. 이때, 다음과 같은 과정을 거쳐서 판단해야 합니다. 먼저, 근거로 제시된 전제들을 자세히 읽어보어야 합니다. 이어서, 제시된 전제들이 법률 가설 판단에 충분한 도움이 된다면, 그 전제의 내용을 이용해 법률 가설의 참 거짓을 판단하십시오. 만약 제시된 전제들이 도움이 되지 않는다면, 당신이 보유한 법률 전문 지식과 추론 능력을 이용해 법률 가설의 참 거짓을 판단하십시오. 법률 가설의 참 거짓을 판단할 때는 반드시 적절한 법리, 판결문, 논리적 구성 요소에 따라 판단하십시오.
        ---
    
        항상 다음과 같은 형식을 따라야 합니다.
    
        법률 가설 : "법률, 규칙, 법리, 범죄 상황 등과 관련한 주장"
    
        전제 : "5개의 전제 사실"
        전제 i: "i번째 전제 사실, 판결문, 법률 조항, 범죄 수사학 교과서 등의 정보로써 법률 가설 판단에 도움이 될 것으로 예상됨"
    
        추론: "법률 가설의 판단을 위한 추론 과정"
    
        답변: "법률 가설의 판단. 참 또는 거짓."
    
        ---
    
        법률 가설: {x['hypothesis']}
    
        전제: {x['premise']}
    
        추론 결과:{x['output']}<|endoftext|>"""})

                data.save_to_disk(f"{self.path_ftdata}/6s_rationales_{key_split}_{self.subsample}.hf")

            elif key_split == 'test':
                data = Dataset.from_pandas(pd.DataFrame(data=data_instance_test[key_split],
                                                        columns=COLUMNS).sample(frac=self.subsample))
                data.save_to_disk(f"{self.path_ftdata}/6s_rationales_{key_split}_{self.subsample}.hf")

    def make_simple_finetuning_instruction_dataset(self):
        COLUMNS = ['hypothesis', 'premise', 'output', 'year', 'hypothesis_id', 'subject',
                   'hypothesis_answer']  # 희두 : 'expert_correction' 컬럼 추가됨(CI_hypothesis_v1.json)
        """
        2013~2020 년도에 해당하는 가설데이터에 대해서만 hypothesis, expert explain 정보를 추출해서 instruction formatted data로 변형함
        """
        with open(self.path_rawdata + ".json", 'r') as f:
            data = f.read()
        raw_dataset = json.loads(data)

        data_instances = dict(train=[], dev=[], test=[])
        data_instances_test = dict(train=[], dev=[], test=[])
        for key in raw_dataset:
            year = int(raw_dataset[key]['year'])
            key_split = None
            gpt_rationales = raw_dataset[key]['gpt_rationales']

            if year in self.years_train:
                key_split = 'train'
            elif year in self.years_dev:
                key_split = 'dev'
            elif year in self.years_test:
                key_split = 'test'
            else:
                raise

            for n in [f'gpt4_rationale_{i}' for i in range(1, 7)]:
                dict_data = dict()
                dict_data['hypothesis'] = raw_dataset[key]['hypothesis']
                dict_data['premise'] = raw_dataset[key]['premise']
                dict_data['output'] = raw_dataset[key]['gpt_rationales'][n]['text']
                # dict_data['output_corrected']  = None
                dict_data['year'] = year
                dict_data['hypothesis_id'] = raw_dataset[key]['hypothesis_id'] + '_' + n
                dict_data['subject'] = raw_dataset[key]['subject']
                dict_data['hypothesis_answer'] = raw_dataset[key]['hypothesis_answer']
                if raw_dataset[key]['gpt_rationales'][n]['gpt4_is_correct'] == 'correct':
                    data_instances[key_split].append(dict_data)

            try:
                dict_data = dict()
                dict_data['hypothesis'] = raw_dataset[key]['hypothesis']
                dict_data['premise'] = raw_dataset[key]['premise']
                dict_data['output'] = raw_dataset[key]['expert_correction']
                # dict_data['output_corrected']  = None
                dict_data['year'] = year
                dict_data['hypothesis_id'] = raw_dataset[key]['hypothesis_id'] + '_expert_corrected'
                dict_data['subject'] = raw_dataset[key]['subject']
                dict_data['hypothesis_answer'] = raw_dataset[key]['hypothesis_answer']
                data_instances[key_split].append(dict_data)
                data_instances_test[key_split].append(dict_data)
            except:
                pass

        for key_split in ['train', 'dev', 'test']:
            if key_split == 'train' or key_split == 'dev':
                data = Dataset.from_pandas(pd.DataFrame(data=data_instances[key_split],
                                                        columns=COLUMNS).sample(frac=self.subsample))
                data = data.map(
                    lambda x: {'text':
                                   f"""주어진 전제에 의할 때, 법률 가설은 참인가 거짓인가?
                                   
                                   전제: {x['premise']}
                                   
                                   법률 가설: {x['hypothesis']}
                                   
                                   추론 결과:{x['output']}<|endoftext|>"""})

                data.save_to_disk(f"{self.path_ftdata}/simple_{key_split}_{self.subsample}.hf")

            elif key_split == 'test':
                data = Dataset.from_pandas(pd.DataFrame(data=data_instances_test[key_split],
                                                        columns=COLUMNS).sample(frac=self.subsample))
                data.save_to_disk(f"{self.path_ftdata}/simple_{key_split}_{self.subsample}.hf")

    def make_6s_solution_instruction_dataset(self):
        COLUMNS = ['hypothesis', 'premise', 'output', 'year', 'hypothesis_id', 'subject',
                   'hypothesis_answer']
        """
        2013~2020 년도에 해당하는 가설데이터에 대해서만 hypothesis, expert explain 정보를 추출해서 instruction formatted data로 변형함
        """
        with open(self.path_rawdata + ".json", 'r') as f:
            data = f.read()
        raw_dataset = json.loads(data)

        data_instances = dict(train=[], dev=[], test=[])
        data_instances_test = dict(train=[], dev=[], test=[])
        data_instances_explain = dict(train=[], dev=[], test=[])

        for key in raw_dataset:
            year = int(raw_dataset[key]['year'])
            key_split = None
            gpt_rationales = raw_dataset[key]['gpt_rationales']

            if year in self.years_train:
                key_split = 'train'
            elif year in self.years_dev:
                key_split = 'dev'
            elif year in self.years_test:
                key_split = 'test'
            else:
                raise

            for n in [f'gpt4_rationale_{i}' for i in range(1, 7)]:
                dict_data = dict()
                dict_data['hypothesis'] = raw_dataset[key]['hypothesis']
                dict_data['premise'] = raw_dataset[key]['premise']
                dict_data['output'] = raw_dataset[key]['gpt_rationales'][n]['text']
                # dict_data['output_corrected']  = None
                dict_data['year'] = year
                dict_data['hypothesis_id'] = raw_dataset[key]['hypothesis_id'] + '_' + n
                dict_data['subject'] = raw_dataset[key]['subject']
                dict_data['hypothesis_answer'] = raw_dataset[key]['hypothesis_answer']
                if raw_dataset[key]['gpt_rationales'][n]['gpt4_is_correct'] == 'correct':
                    data_instances[key_split].append(dict_data)

            try:
                dict_data = dict()
                dict_data['hypothesis'] = raw_dataset[key]['hypothesis']
                dict_data['premise'] = raw_dataset[key]['premise']
                dict_data['output'] = raw_dataset[key]['expert_correction']
                # dict_data['output_corrected']  = None
                dict_data['year'] = year
                dict_data['hypothesis_id'] = raw_dataset[key]['hypothesis_id'] + '_expert_corrected'
                dict_data['subject'] = raw_dataset[key]['subject']
                dict_data['hypothesis_answer'] = raw_dataset[key]['hypothesis_answer']
                data_instances_test[key_split].append(dict_data)
            except:
                pass

            if raw_dataset[key]['expert_explain'] != 'nan' and raw_dataset[key]['expert_explain'] != 'None':
                dict_data = dict()
                dict_data['hypothesis'] = raw_dataset[key]['hypothesis']
                dict_data['premise'] = raw_dataset[key]['premise']
                dict_data['output'] = raw_dataset[key]['expert_explain']
                dict_data['year'] = year
                dict_data['hypothesis_id'] = raw_dataset[key]['hypothesis_id'] + '_' + '_expert_explain'
                dict_data['subject'] = raw_dataset[key]['subject']
                dict_data['hypothesis_answer'] = raw_dataset[key]['hypothesis_answer']
                data_instances_explain[key_split].append(dict_data)

        for key_split in ['train', 'dev', 'test']:
            if key_split == 'train' or key_split == 'dev':
                data = Dataset.from_pandas(pd.DataFrame(data=data_instances[key_split],
                                                        columns=COLUMNS).sample(frac=self.subsample))
                data_explain = Dataset.from_pandas(pd.DataFrame(data=data_instances_explain[key_split],
                                                                columns=COLUMNS).sample(frac=self.subsample))

                data = data.map(
                    lambda x: {'text':
                                   f"""당신은 범죄 수사 전문가로, 주어진 법률 가설의 참, 거짓을 판단할 수 있습니다. 당신의 임무는 주어진 법률 가설과 전제를 검토하여, 법률 가설이 참인지 거짓인지를 판단하는 것입니다. 이때, 다음과 같은 과정을 거쳐서 판단해야 합니다. 먼저, 근거로 제시된 전제들을 자세히 읽어보어야 합니다. 이어서, 제시된 전제들이 법률 가설 판단에 충분한 도움이 된다면, 그 전제의 내용을 이용해 법률 가설의 참 거짓을 판단하십시오. 만약 제시된 전제들이 도움이 되지 않는다면, 당신이 보유한 법률 전문 지식과 추론 능력을 이용해 법률 가설의 참 거짓을 판단하십시오. 법률 가설의 참 거짓을 판단할 때는 반드시 적절한 법리, 판결문, 논리적 구성 요소에 따라 판단하십시오.
         ---

         항상 다음과 같은 형식을 따라야 합니다.

         법률 가설 : "법률, 규칙, 법리, 범죄 상황 등과 관련한 주장"

         전제 : "5개의 전제 사실"
         전제 i: "i번째 전제 사실, 판결문, 법률 조항, 범죄 수사학 교과서 등의 정보로써 법률 가설 판단에 도움이 될 것으로 예상됨"

         추론: "법률 가설의 판단을 위한 추론 과정"

         답변: "법률 가설의 판단. 참 또는 거짓."

         ---

         법률 가설: {x['hypothesis']}

         전제: {x['premise']}

         추론 결과:{x['output']}<|endoftext|>"""})

                data_explain = data_explain.map(
                    lambda x: {'text':
                                   f"""당신은 범죄 수사 전문가로, 주어진 법률 가설의 참, 거짓을 판단할 수 있습니다. 당신의 임무는 주어진 법률 가설을 검토하여, 법률 가설이 참인지 거짓인지를 판단하는 것입니다. 이때, 당신이 보유한 법률 전문 지식과 추론 능력을 이용해 법률 가설의 참 거짓을 판단하십시오. 법률 가설의 참 거짓을 판단할 때는 반드시 적절한 법리, 판결문, 논리적 구성 요소에 따라 진행하십시오.
                ---

                항상 다음과 같은 형식을 따라야 합니다.

                법률 가설 : "법률, 규칙, 법리, 범죄 상황 등과 관련한 주장"

                추론: "법률 가설의 판단을 위한 추론 과정"

                답변: "법률 가설의 판단. 참 또는 거짓."

                ---

                법률 가설 : {x['hypothesis']}
                추론 결과 : {x['output']}<|endoftext|>"""})

                dataset_cc = concatenate_datasets([data, data_explain])

                dataset_cc.save_to_disk(f"{self.path_ftdata}/6s_solution_{key_split}_{self.subsample}.hf")

            elif key_split == 'test':
                data_test = Dataset.from_pandas(pd.DataFrame(data=data_instances_test[key_split],
                                                             columns=COLUMNS).sample(frac=self.subsample))

                data_test.save_to_disk(f"{self.path_ftdata}/6s_solution_{key_split}_{self.subsample}.hf")