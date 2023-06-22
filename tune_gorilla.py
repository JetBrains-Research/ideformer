import os

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,6"
os.environ['CUDA_PATH']='/usr/local/cuda-11'

import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset
import re

def prep_gorilla(addr):
    # manually preparing the dataset
    df = pd.read_json(addr, lines=True)
    df_instructions_list = df.code.apply(lambda x: re.findall(r'###.?Instruction: (.*?)\n', x))
    df_outputs_list = df.code.apply(lambda x: re.findall(r'###.?Output: (.+)', x, flags=re.DOTALL))
    
    both_ok = (df_outputs_list.apply(len) == 1) & (df_instructions_list.apply(len) == 1)
    
    df = df[both_ok]
    
    df['instruction'] = df_instructions_list[both_ok].apply(lambda x: x[0])
    df['output'] = df_outputs_list[both_ok].apply(lambda x: x[0])
    
    return Dataset.from_pandas(df.astype(str))

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

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer


def train():
    # prepare datasets
    tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")
    tokenizer.add_special_tokens({'pad_token': '<?>'})

    def tokenize_function(examples):
        return tokenizer(examples['instruction'], text_target=examples['output'], truncation=True, 
                        padding='max_length', max_length=640)


    hf_train_dset = prep_gorilla('/mnt/data/mart/gorilla/data/apibench/huggingface_train.json')
    hf_eval_dset = prep_gorilla('/mnt/data/mart/gorilla/data/apibench/huggingface_eval.json')

    tok_hf_train_dset = hf_train_dset.map(tokenize_function, batched=True)
    tok_hf_eval_dset = hf_eval_dset.map(tokenize_function, batched=True)

    tok_hf_train_dset = tok_hf_train_dset.remove_columns(['code', 
                                                        'api_call', 
                                                        'provider',
                                                        'api_data',
                                                        'instruction',
                                                        'output',
                                                        '__index_level_0__'])

    tok_hf_eval_dset = tok_hf_eval_dset.remove_columns(['code', 
                                                        'api_call', 
                                                        'provider',
                                                        'api_data',
                                                        'instruction',
                                                        'output',
                                                        '__index_level_0__'])


    # prepare model
    config = AutoConfig.from_pretrained('tiiuae/falcon-7b', trust_remote_code=True)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        
    model.tie_weights()

    device_map = infer_auto_device_map(model, max_memory={0: "5GiB", 1: "5GiB", 2: "5GiB", 3: "5GiB", 4: "5GiB", 5: "5GiB"}, # stop doing it manually, since now it's equal
                                    no_split_module_classes=["DecoderLayer"])

    model = load_checkpoint_and_dispatch(model, 
                                        "/mnt/data/mart/falcon-7b-sharded-bf16", 
                                        device_map=device_map)

    print(model.hf_device_map)


    # prepare trainer
    training_args = Seq2SeqTrainingArguments(output_dir="/mnt/data/mart/test_trainer", learning_rate=2e-5, 
                                            num_train_epochs=5, warmup_ratio=0.03, 
                                            gradient_accumulation_steps=64, save_strategy='epoch',
                                            load_best_model_at_end=True,
                                            per_device_train_batch_size=1, evaluation_strategy='epoch')

    trainer = Seq2SeqTrainer(model=model, args=training_args, 
                            train_dataset=tok_hf_train_dset, 
                            eval_dataset=tok_hf_eval_dset, )


    # GO!
    trainer.train()

if __name__ == "__main__":
    train()
