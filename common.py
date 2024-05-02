__all__ = [
    'CHATML_CHAT_TEMPLATE',
    'ROLES_MAP',
    'model_id',
    'config',
    'tokenizer',
    'convert_open_orca_slim_orca_to_chatml_message',
    'convert_open_orca_slim_orca_to_chatml_dataset',
    'convert_open_orca_1million_gpt_4_to_chatml_message',
    'convert_open_orca_1million_gpt_4_to_chatml_dataset',
    'convert_huggingfaceh4_ultrachat_200k_to_chatml_dataset',
    'convert_teknium_open_hermes_2_5_to_chatml_dataset',
    'convert_hf_dataset_to_chatml_dataset',
]

import numpy as np
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
from datasets import load_dataset

CHATML_CHAT_TEMPLATE = (
    "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)

ROLES_MAP = {
    'system': 'system',
    'system_prompt': 'system', # Open-Orca/1million-gpt-4
    
    'user': 'user',
    'human': 'user', # Open-Orca/SlimOrca
    'question': 'user', # Open-Orca/1million-gpt-4

    'assistant': 'assistant',
    'gpt': 'assistant', # Open-Orca/SlimOrca
    'response': 'assistant', # Open-Orca/1million-gpt-4
}


model_id = 'cognitivecomputations/dolphin-2.6-mistral-7b'
config = AutoConfig.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
tokenizer.chat_template = CHATML_CHAT_TEMPLATE


#
# Open-Orca/SlimOrca
#
def convert_open_orca_slim_orca_to_chatml_message(c: dict[str, str]) -> dict[str, str]:
    return {
        'role': ROLES_MAP[c['from']],
        'content': c['value'],
    }


def convert_open_orca_slim_orca_to_chatml_dataset(ds) -> list[list[dict[str, str]]]:
    output: list[list[dict[str, str]]] = []

    for n in tqdm(ds['train']):
        cs: list[dict[str, str]] = n['conversations']
        
        messages: list[dict[str, str]] = [
            convert_open_orca_slim_orca_to_chatml_message(c)
            for c in cs
        ]

        output.append(messages)

    return output

#
# Open-Orca/1million-gpt-4
#
def convert_open_orca_1million_gpt_4_to_chatml_messages(c: dict[str, str]) -> list[dict[str, str]]:
    return [
        {
            'role': 'system',
            'content': c['system_prompt'],
        },
        {
            'role': 'user',
            'content': c['question'],
        },
        {
            'role': 'assistant',
            'content': c['response'],
        },
    ]


def convert_open_orca_1million_gpt_4_to_chatml_dataset(ds) -> list[list[dict[str, str]]]:
    output: list[list[dict[str, str]]] = []

    for n in tqdm(ds['train']):
        messages: list[dict[str, str]] = convert_open_orca_1million_gpt_4_to_chatml_messages(n)
        output.append(messages)

    return output

#
# HuggingFaceH4/ultrachat_200k
#
def convert_huggingfaceh4_ultrachat_200k_to_chatml_dataset(ds) -> list[list[dict[str, str]]]:
    output: list[list[dict[str, str]]] = ds['train_sft']['messages'] + ds['train_gen']['messages']
    return output

#
# teknium/OpenHermes-2.5
#
def convert_teknium_open_hermes_2_5_to_chatml_message(c: dict[str, str]) -> dict[str, str]:
    return {
        'role': ROLES_MAP[c['from']],
        'content': c['value'],
    }

def convert_teknium_open_hermes_2_5_to_chatml_dataset(ds) -> list[list[dict[str, str]]]:
    output: list[list[dict[str, str]]] = []

    for n in tqdm(ds['train']):
        cs: list[dict[str, str]] = n['conversations']
        
        messages: list[dict[str, str]] = [
            convert_open_orca_slim_orca_to_chatml_message(c)
            for c in cs
        ]

        output.append(messages)

    return output

#
# cognitivecomputations/dolphin
#
def convert_cognitivecomputations_dolphin_to_chatml_messages(c: dict[str, str]) -> list[dict[str, str]]:
    return [
        {
            'role': 'system',
            'content': c['instruction'],
        },
        {
            'role': 'user',
            'content': c['input'],
        },
        {
            'role': 'assistant',
            'content': c['output'],
        },
    ]


def convert_cognitivecomputations_dolphin_to_chatml_dataset(ds) -> list[list[dict[str, str]]]:
    output: list[list[dict[str, str]]] = []

    for n in tqdm(ds['train']):
        messages: list[dict[str, str]] = convert_cognitivecomputations_dolphin_to_chatml_messages(n)
        output.append(messages)

    return output

#
# convert_hf_dataset_to_chatml_dataset
#
def convert_hf_dataset_to_chatml_dataset(dataset: str) -> list[list[dict[str, str]]]:
    chatml_ds: list[list[dict[str, str]]] = None
    
    if dataset == 'Open-Orca/SlimOrca':
        ds = load_dataset(dataset)
        chatml_ds = convert_open_orca_slim_orca_to_chatml_dataset(ds)
    elif dataset == 'Open-Orca/1million-gpt-4':
        ds = load_dataset(dataset)
        chatml_ds = convert_open_orca_1million_gpt_4_to_chatml_dataset(ds)
    elif dataset == 'HuggingFaceH4/ultrachat_200k':
        ds = load_dataset(dataset)
        chatml_ds = convert_huggingfaceh4_ultrachat_200k_to_chatml_dataset(ds)
    elif dataset == 'teknium/OpenHermes-2.5':
        ds = load_dataset(dataset)
        chatml_ds = convert_teknium_open_hermes_2_5_to_chatml_dataset(ds)
    elif dataset == 'cognitivecomputations/dolphin':
        chatml_ds = []

        data_files = [
            'flan1m-alpaca-uncensored-deduped.jsonl',
            'flan5m-alpaca-uncensored-deduped.jsonl',
        ]

        for n in data_files:
            ds = load_dataset(dataset, data_files=n)
            chatml_ds.extend(convert_cognitivecomputations_dolphin_to_chatml_dataset(ds))
    else:
        raise ValueError(f'Unknown dataset {dataset}')

    return chatml_ds