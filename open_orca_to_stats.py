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

def bwt(s: str) -> str:
    """Apply Burrows–Wheeler transform to input string."""
    assert "\002" not in s and "\003" not in s, "Input string cannot contain STX and ETX characters"
    s = "\002" + s + "\003"  # Add start and end of text marker
    table = sorted(s[i:] + s[:i] for i in range(len(s)))  # Table of rotations of string
    last_column = [row[-1:] for row in table]  # Last characters of each row
    return "".join(last_column)  # Convert list of characters into string

def ibwt(r: str) -> str:
    """Apply inverse Burrows–Wheeler transform."""
    table = [""] * len(r)  # Make empty table
    for _ in range(len(r)):
        table = sorted(r[i] + table[i] for i in range(len(r)))  # Add a column of r

    s = next((row for row in table if row.endswith("\003")), "") # Iterate over and check whether last character ends with ETX or not
    return s.rstrip("\003").strip("\002")  # Retrive data from array and get rid of start and end markers

if __name__ == '__main__':
    for i, n in enumerate(open_orca_ds['train']):
        system = n['system_prompt']
        user = n['question']
        assistant = n['response']
        
        # prompt = format_prompt(system, user, assistant)
        # print('# prompt:', len(prompt))
        # print(f'prompt: {prompt!r}')
        
        # tokens_ids = tokenizer_encode_conversation(prompt)
        # print('# tokens_ids:', len(tokens_ids))
        # print(f'tokens_ids: {tokens_ids!r}')
        
        # prompt_tokens = tokenizer_decode_conversation(tokens_ids)
        # print('# prompt_tokens:', len(prompt_tokens))
        # print(f'prompt_tokens: {prompt_tokens!r}')

        # tokens_ids_unique = list(OrderedSet(tokens_ids))
        # print('# tokens_ids_unique:', len(tokens_ids_unique))
        # print(f'tokens_ids_unique: {tokens_ids_unique!r}')

        # prompt_tokens_unique = tokenizer_decode_conversation(tokens_ids_unique)
        # print('# prompt_tokens_unique:', len(prompt_tokens_unique))
        # print(f'prompt_tokens_unique: {prompt_tokens_unique!r}')

        # prompt_tokens_unique_str = ''.join(prompt_tokens_unique)
        # print('# prompt_tokens_unique_str:', len(prompt_tokens_unique_str))
        # print(f'prompt_tokens_unique_str: {prompt_tokens_unique_str!r}')

        # prompt_tokens_unique_str_clean = stopwords.clean([n.lower() for n in prompt_tokens_unique], 'en')
        # print('# prompt_tokens_unique_str_clean:', len(prompt_tokens_unique_str_clean))
        # print(f'prompt_tokens_unique_str_clean: {prompt_tokens_unique_str_clean!r}')

        # prompt_tokens_unique_str_clean = ''.join(prompt_tokens_unique_str_clean)
        # print('# prompt_tokens_unique_str_clean:', len(prompt_tokens_unique_str_clean))
        # print(f'prompt_tokens_unique_str_clean: {prompt_tokens_unique_str_clean!r}')

        # prompt_tokens_unique_btw = bwt(prompt_tokens_unique_str_clean)
        # print('# prompt_tokens_unique_btw:', len(prompt_tokens_unique_btw))
        # print(f'prompt_tokens_unique_btw: {prompt_tokens_unique_btw!r}')

        # prompt_tokens_unique_bz2 = bz2.compress(prompt_tokens_unique_str_clean.encode())
        # print('# prompt_tokens_unique_bz2:', len(prompt_tokens_unique_bz2))
        # print(f'prompt_tokens_unique_bz2: {prompt_tokens_unique_bz2!r}')

        # prompt_tokens_unique_gzip = gzip.compress(prompt_tokens_unique_str_clean.encode())
        # print('# prompt_tokens_unique_gzip:', len(prompt_tokens_unique_gzip))
        # print(f'prompt_tokens_unique_gzip: {prompt_tokens_unique_gzip!r}')

        # print('-' * 32)

        prompt = format_prompt(system, user, assistant)
        print('# prompt:', len(prompt))
        print(f'prompt: {prompt!r}')

        prompt_ordered = list(OrderedSet(prompt))
        print('# prompt_ordered:', len(prompt_ordered))
        print(f'prompt_ordered: {prompt_ordered!r}')

        prompt_ordered_str = ''.join(prompt_ordered)
        print('# prompt_ordered_str:', len(prompt_ordered_str))
        print(f'prompt_ordered_str: {prompt_ordered_str!r}')

        prompt_ordered_str_bz2 = bz2.compress(prompt_ordered_str.encode())
        print('# prompt_ordered_str_bz2:', len(prompt_ordered_str_bz2))
        print(f'prompt_ordered_str_bz2: {prompt_ordered_str_bz2!r}')

        prompt_ordered_str_gzip = gzip.compress(prompt_ordered_str.encode())
        print('# prompt_ordered_str_gzip:', len(prompt_ordered_str_gzip))
        print(f'prompt_ordered_str_gzip: {prompt_ordered_str_gzip!r}')

        print('=' * 32)

        break