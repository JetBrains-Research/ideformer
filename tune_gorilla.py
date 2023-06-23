import os

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,6"
os.environ['CUDA_PATH']='/usr/local/cuda-11'

import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset
import re

def prep_dataset(addr, tokenizer):
    # prepare datasets

    def tokenize_function(examples):
        return tokenizer( [f"<user>: {instr}\n<IDE-genie>: {answ}" for instr, answ in zip(examples['instruction'], examples['output'])]
                        # examples['instruction'], text_target=examples['output'], 
                        # truncation=True, padding='max_length', max_length=640
                        )

    # manually preparing the dataset
    df = pd.read_json(addr, lines=True)
    df_instructions_list = df.code.apply(lambda x: re.findall(r'###.?Instruction: (.*?)\n', x))
    df_outputs_list = df.code.apply(lambda x: re.findall(r'###.?Output: (.+)', x, flags=re.DOTALL))
    
    both_ok = (df_outputs_list.apply(len) == 1) & (df_instructions_list.apply(len) == 1)
    
    df = df[both_ok]
    
    df['instruction'] = df_instructions_list[both_ok].apply(lambda x: x[0])
    df['output'] = df_outputs_list[both_ok].apply(lambda x: x[0])

    dset = Dataset.from_pandas(df.astype(str))

    dset = dset.map(tokenize_function, batched=True)

    dset = dset.remove_columns(['code', 'api_call', 'provider','api_data','instruction','output','__index_level_0__'])
    
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

    device_map = infer_auto_device_map(model, max_memory={0: "5GiB",1: "5GiB",2: "5GiB",3: "5GiB",4: "5GiB",5: "5GiB"}, 
                                       no_split_module_classes=["DecoderLayer"])

    model = load_checkpoint_and_dispatch(model, 
                                         model_addr, 
                                         device_map=device_map)

    return model

from transformers import TrainingArguments, Trainer


def train():
    tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")
    tokenizer.add_special_tokens({'pad_token': '<?>'})

    hf_train_dset = prep_dataset('/mnt/data/mart/gorilla/data/apibench/huggingface_train.json', tokenizer)
    hf_eval_dset = prep_dataset('/mnt/data/mart/gorilla/data/apibench/huggingface_eval.json', tokenizer)

    model = prep_model("/mnt/data/mart/falcon-7b-sharded-bf16") 

    # prepare trainer
    training_args = TrainingArguments(output_dir="/mnt/data/mart/test_trainer", 
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
    train()
