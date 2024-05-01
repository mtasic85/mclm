import numpy as np
from transformers import AutoConfig, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


config = AutoConfig.from_pretrained('cognitivecomputations/dolphin-2.6-mistral-7b')

CHATML_CHAT_TEMPLATE = (
    "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)

tokenizer = AutoTokenizer.from_pretrained('cognitivecomputations/dolphin-2.6-mistral-7b', trust_remote_code=True, use_fast=True)
tokenizer.chat_template = CHATML_CHAT_TEMPLATE

open_orca_slim_orca_ds = load_dataset('Open-Orca/SlimOrca')

OPEN_ORCA_SLIM_ORCA_ROLES_MAP = {
    'system': 'system',
    'user': 'user',
    'human': 'user',
    'assistant': 'assistant',
    'gpt': 'assistant',
}


def tokenizer_encode_conversation(messages: list[dict[str, str]], add_generation_prompt=False) -> list[int]:    
    tokens_ids: list[int] = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_generation_prompt)
    return tokens_ids


def create_markov_chain_transition_matrix_for_open_orca_slim_orca_dataset(cs: list[dict[str, str]]) -> np.ndarray:
    ds: dict = open_orca_slim_orca_ds
    vocab_size: int = config.vocab_size
    mctm = np.zeros((vocab_size, vocab_size), dtype=np.uint32)
    tokens_ids: list[int] = tokenizer_encode_conversation(cs, add_generation_prompt=True)

    for i in range(0, len(tokens_ids) - 1):
        curr_token = tokens_ids[i]
        next_token = tokens_ids[i + 1]
        mctm[(curr_token, next_token)] += 1

    return mctm


if __name__ == '__main__':
    ptm_mctm: np.ndarray = np.load('open-orca-slim-orca-mctm.npy')
    print(ptm_mctm)
    print(np.sum(ptm_mctm))
    print(np.sum(ptm_mctm != 0))

    messages = [
        {'role': 'system', 'content': 'You are an AI assistant.'},
        {'role': 'user', 'content': 'What do you know about Australia?'},
    ]

    itm_mctm: np.ndarray = create_markov_chain_transition_matrix_for_open_orca_slim_orca_dataset(messages)
    print(itm_mctm)
    print(np.sum(itm_mctm))
    print(np.sum(itm_mctm != 0))

    atm: np.ndarray = ptm_mctm * (itm_mctm != 0)
    print(atm)
    print(np.sum(atm))
    print(np.sum(atm != 0))
