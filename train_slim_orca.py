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


def tokenizer_encode_conversation(cs: list[dict[str, str]]) -> list[int]:    
    messages: list[dict[str, str]] = [
        {
            'role': OPEN_ORCA_SLIM_ORCA_ROLES_MAP[c['from']],
            'content': c['value'],
        }
        for c in cs
    ]

    tokens_ids: list[int] = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    return tokens_ids


def create_markov_chain_transition_matrix_from_open_orca_slim_orca_dataset() -> np.ndarray:
    ds: dict = open_orca_slim_orca_ds
    vocab_size: int = config.vocab_size
    mctm = np.zeros((vocab_size, vocab_size), dtype=np.uint32)
    
    for n in tqdm(ds['train']):
        cs: list[dict[str, str]] = n['conversations']
        tokens_ids: list[int] = tokenizer_encode_conversation(cs)

        for i in range(0, len(tokens_ids) - 1):
            curr_token = tokens_ids[i]
            next_token = tokens_ids[i + 1]
            mctm[(curr_token, next_token)] += 1

    return mctm


if __name__ == '__main__':
    mctm: np.ndarray = create_markov_chain_transition_matrix_from_open_orca_slim_orca_dataset()
    print(mctm)
    np.save('open-orca-slim-orca-mctm.npy', mctm)
