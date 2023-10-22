import bz2
import gzip

import numpy as np
from transformers import AutoTokenizer, MistralConfig
from datasets import load_dataset
from tqdm import tqdm
from ordered_set import OrderedSet
import stopwords

configuration = MistralConfig()
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1')

open_orca_ds = load_dataset('Open-Orca/OpenOrca')
print(open_orca_ds)

def format_prompt(system: str, user: str, assistant: str) -> str:
    prompt = f'''<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
{assistant}<|im_end|>'''
    return prompt

def tokenizer_encode_conversation(prompt: str) -> list[int]:
    tokens_ids: list[int] = tokenizer.encode(prompt)
    return tokens_ids

def tokenizer_decode_conversation(tokens_ids: list[int]) -> list[str]:
    prompt_tokens: list[str] = tokenizer.batch_decode(tokens_ids)
    return prompt_tokens

if __name__ == '__main__':
    # for i, n in tqdm(enumerate(open_orca_ds['train'])):
    for n in tqdm(open_orca_ds['train']):
        system = n['system_prompt']
        user = n['question']
        assistant = n['response']

        prompt = format_prompt(system, user, assistant)
        # print('# prompt:', len(prompt))
        # print(f'prompt: {prompt!r}')

        prompt_ordered = list(OrderedSet(prompt))
        # print('# prompt_ordered:', len(prompt_ordered))
        # print(f'prompt_ordered: {prompt_ordered!r}')

        prompt_ordered_str = ''.join(prompt_ordered)
        # print('# prompt_ordered_str:', len(prompt_ordered_str))
        # print(f'prompt_ordered_str: {prompt_ordered_str!r}')

        # print('=' * 32)
        
        # print(len(prompt), len(prompt_ordered_str), len(prompt_ordered_str) / len(prompt))
