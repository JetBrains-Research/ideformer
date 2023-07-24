import os

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,6"
# os.environ['CUDA_PATH']='/usr/local/cuda-11'

import pandas as pd
import yaml
import numpy as np
from datasets import concatenate_datasets, Dataset
import re
import argparse # move to hydra later, I guess...

with open('categories.yml') as f:
    categories_list = yaml.safe_load(f)
cat_description = '; '.join([c['description'] for c in categories_list])
categories_dict = {c['name']:c for c in categories_list}

with open('tools_docs.yml') as f:
    tools_short_doc = yaml.safe_load(f)

with open('tools_cheatsheets.yml') as f:
    tools_long_doc = yaml.safe_load(f)

def make_simple_prediction(record):
    return f'<user>: {record.invocation}\n'\
           f'<IDE-genie>: {record.cmd}<|endoftext|>'

def make_cmd_cat_prediction(record):
    selected_categories = ", ".join(record.cmdcath)
    
    return f'<user>: {record.invocation}\n'\
           f'<category encyclopedia>: {cat_description}\n'\
           f'<logic>[Medium:success]: <|{selected_categories}|><|endoftext|>'

def make_cmd_tools_prediction(record):
    tools_proposal = []
    for c in record.cmdcath:
        tools_proposal += categories_dict[c]['utils']
    tools_description = '; '.join([f'{t}: {tools_short_doc[t]}' for t in tools_proposal])
    selected_tools = ", ".join(record.cmdset)
    
    selected_categories = ", ".join(record.cmdcath)
    return f'<user>: {record.invocation}\n'\
           f'<category logic>[trivial:success]: <|{selected_categories}|>\n'\
           f'<tools encyclopedia>: {tools_description}\n'\
           f'<tool logic>[medium:success]: <|{selected_tools}|><|endoftext|>'

def make_cmd_usage_prediction(record):
    tools_cheatsheets = []
    tools_description = '\n\n'.join([f'{tools_long_doc[t]}' for t in record.cmdset])
    
    selected_tools = ", ".join(record.cmdset)
    selected_categories = ", ".join(record.cmdcath)
    return f'<user>: {record.invocation}\n'\
           f'<category logic>[trivial:success]: <|{selected_categories}|>\n'\
           f'<tool logic>[medium:success]: <|{selected_tools}|>\n'\
           f'<cheatsheets>: {tools_description}\n'\
           f'<IDE-genie>: {record.cmd}<|endoftext|>' 

def prep_bash_cmd_dataset(addr, tokenizer, converter_functions=['make_simple_prediction']):
    tokenize_functions = [lambda records: [tokenizer() for r in records] for conv_f in converter_functions]

    df = pd.read_json(addr, lines=True)
    
    for c_f in converter_functions:
        df[c_f] = df.apply(globals()[c_f], axis=1)
        
    dsets = []
    for c_f in converter_functions:
        dset = Dataset.from_pandas(df)
        dset = dset.map(lambda r: tokenizer(r[c_f]))
        dsets.append(dset)

    dset = concatenate_datasets(dsets)
    
    return dset

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaForCausalLM,
    T5Tokenizer,
)


from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForCausalLM

def prep_model(model_addr):
    # prepare model
    config = AutoConfig.from_pretrained('tiiuae/falcon-7b', trust_remote_code=True)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    model.tie_weights()

    # device_map = infer_auto_device_map(model, max_memory={0: "5GiB",1: "5GiB",2: "5GiB",3: "5GiB",4: "5GiB",5: "5GiB"}, 
    #                                    no_split_module_classes=["DecoderLayer"])

    model = load_checkpoint_and_dispatch(model, 
                                         model_addr, 
                                         device_map="auto",  no_split_module_classes=["DecoderLayer"])

    return model

from transformers import TrainingArguments, Trainer

from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(version_base=None, config_path=".", config_name="config")
def train(cfg):
    tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")
    tokenizer.add_special_tokens({'pad_token': '<?>'})

    hf_train_dset = prep_bash_cmd_dataset(cfg.dataset.train, tokenizer, cfg.converter_functions)
    hf_eval_dset = prep_bash_cmd_dataset(cfg.dataset.test, tokenizer, cfg.converter_functions)

    model = prep_model(cfg.model_addr) 

    # prepare trainer
    training_args = TrainingArguments(output_dir=cfg.save_location, 
                                            learning_rate=2e-5, 
                                            per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=64,
                                            num_train_epochs=5, 
                                            warmup_ratio=0.03, 
                                            save_strategy='epoch', evaluation_strategy='epoch',
                                            load_best_model_at_end=True,
                                            lr_scheduler_type='cosine',)

    trainer = Trainer(model=model, args=training_args, 
                            train_dataset=hf_train_dset, 
                            eval_dataset=hf_eval_dset, 
                            data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),)


    # GO!
    trainer.train()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Tune the falcon-7B model for the nl2bash dataset')
    parser.add_argument('--converter_functions', nargs='*', help='which functions to invoke for string preparation, by default invokes only "make_simple_prediction"')
    args = parser.parse_args()
    train(args)
