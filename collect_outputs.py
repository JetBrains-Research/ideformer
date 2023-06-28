import os

os.environ['CUDA_PATH']='/usr/local/cuda-11'

import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset
import re

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch




from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForCausalLM

from tune_gorilla import prep_model


def prepare_pipeline(model_addr):
    tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")
    tokenizer.add_special_tokens({'pad_token': '<?>'})

    model = prep_model(model_addr)
    model = model.eval()
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        # torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    return pipeline, tokenizer

# model = prep_model("/mnt/data/mart/test_trainer/checkpoint-625/")
# model = prep_model("/mnt/data/mart/falcon-7b-sharded-bf16/")
# model = prep_model("/mnt/data/mart/test_rewired_checkpoint/")
# model = prep_model("/mnt/data/mart/with_description/")
# model = prep_model("/mnt/data/mart/only_call_stop/")

from tqdm.auto import tqdm

def generate(pipeline, tokenizer, save_addr):
    eval_hf = pd.read_json('/mnt/data/mart/gorilla/data/apibench/huggingface_eval.json', lines=True)
    apis_list = pd.read_json('/mnt/data/mart/gorilla/data/api/huggingface_api.jsonl', lines=True)

    generations = []

    for i, row in tqdm(eval_hf.iterrows(), total=len(eval_hf)):
        cur_gen = {}

        req = re.findall(r'###.?Instruction: (.*)', row.code)[0]
        cur_gen['request'] = req
        cur_gen['expected_call'] = row.api_call

        prompt = f"<user>: {req} \n<IDE-genie>: "
        sequences = pipeline(prompt, max_length=256, do_sample=True, top_k=10, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
        cur_gen['generated_call'] = sequences[0]['generated_text'][len(prompt):]
        generations.append(cur_gen)
    
    pd.DataFrame(generations).to_json(save_addr, orient='records', lines=True)

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect answers of the model for the datase of queries')
    parser.add_argument('--model-checkpoint', type=str, help='checkpoint dir path')
    parser.add_argument('--save-addr', type=str, help='JSON path to save')
    args = parser.parse_args()
    
    pipeline, tokenizer = prepare_pipeline(args.model_checkpoint)
    generate(pipeline, tokenizer, args.save_addr)


